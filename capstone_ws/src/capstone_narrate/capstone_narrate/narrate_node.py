"""capstone_narrate — event-triggered tactical narration on /narration.

Pipeline (per integration plan):
  /tracks (+/fingerprints)  --build_ctx-->  ctx  --classify_ctx-->  bucket
     bucket B  -> Layer 2 guard abstention  (torch-free, synchronous, v1 path)
     bucket A/C-> Layer 1/3 LLM             (lazy GPU load, async worker; DORMANT in v1)

Why this shape:
  * The ctx->bucket ROUTER (`swarm_intent.coverage.classify_ctx`, pinned peer repo
    commit 69ac2b6) is pure/torch-free and FROZEN. We consume it directly.
  * `swarm_intent.pipeline_v2` (which holds the Layer 1/3 model dispatch AND
    `_layer2_abstain`) imports torch at module load, so it is imported ONLY inside
    the worker's model path. In v1 no ctx ever routes to A/C (no STGT `/formation`
    yet => build_ctx emits the degenerate ctx => classify_ctx always returns bucket
    B on the `('unknown','unknown')` pair), so that import — and the GPU — is never
    touched. The abstention Narration is rendered HERE from the router's own
    guard_reasons; source="template", template_id from ctx_builder's meta.

Async model (kept correct even though the model path is dormant in v1):
  MultiThreadedExecutor + a single ReentrantCallbackGroup; subscription callbacks
  only latch the latest message and mark a trigger dirty. A low-rate coalescing
  timer turns bursts of triggers into ONE narration cycle. Bucket B is rendered
  inline (cheap); bucket A/C is offloaded to a 1-thread pool so the executor never
  blocks on a GPU generate(). The lazy loader checks `gpu_free_gib(required_gib)`
  before loading anything (LOCKED CONFIG: report-and-halt, never crash mid-run).

Triggers: a NEW global_id appears on /tracks, k_estimated changes on /fingerprints,
OR (Phase 5, now ENABLED) the STGT formation_class changes on /formation.

Phase-5 activation: with the capstone_formation publisher live, a CONFIDENT
in-vocabulary formation makes build_ctx emit a REAL (non-degenerate) ctx, which
classify_ctx routes to bucket A/C — i.e. the LLM path is no longer dormant. Actual
model loading is still guarded by gpu_free_gib (report-and-halt on a contended
shared GPU), so activation never crashes mid-run; a below-floor or absent formation
still falls back to the torch-free degenerate abstention.
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from capstone_msgs.msg import TrackArray, FingerprintSet, FormationEstimate, Narration
from capstone_common.health import HealthReporter, qos_reliable, qos_latched

# Peer repo (READ-ONLY), pinned commit 69ac2b69015b7d8e37b388acb3443f5073c0eae3.
# activate.sh puts this on PYTHONPATH; the defensive insert here keeps the node
# runnable if someone forgets to re-source. torch-free: only pulls coverage/context_spec.
_AADHYA = os.path.expanduser('~/aadhya/uav-llm')
_AADHYA_SRC = os.path.join(_AADHYA, 'src')
if _AADHYA_SRC not in sys.path:
    sys.path.insert(0, _AADHYA_SRC)

from swarm_intent.coverage import classify_ctx, BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402
from capstone_narrate.ctx_builder import build_ctx  # noqa: E402

# Human text for the ctx-path guard reasons classify_ctx can emit. The KEYS are the
# frozen router vocabulary (coverage.classify_ctx); this rendering is the node's own
# (source="template"). GUARD_REASON_TEXT proper lives in pipeline_v2, which is
# torch-locked, so it is deliberately NOT imported here.
_GUARD_TEXT = {
    'oov_name': 'no in-vocabulary formation is present (STGT classifier offline)',
    'low_confidence': 'every observation window is below the confidence threshold',
    'dominant_history_contradiction': 'the dominant formation is ambiguous',
}


class NarrateNode(Node):
    def __init__(self):
        super().__init__('capstone_narrate')
        self.declare_parameter('passthrough_truth', False)
        self.declare_parameter('keep_warm', False)      # opt-in GPU residency; default lazy
        self.declare_parameter('required_gib', 15.0)    # gpu_free_gib guard threshold
        pt = self.get_parameter('passthrough_truth').value
        self._keep_warm = bool(self.get_parameter('keep_warm').value)
        self._required_gib = float(self.get_parameter('required_gib').value)

        self._cbg = ReentrantCallbackGroup()
        self._pub = self.create_publisher(Narration, '/narration', qos_latched())
        self.health = HealthReporter(self, pt)

        # latched upstream state + trigger bookkeeping
        self._lock = threading.Lock()
        self._last_tracks = None
        self._last_k = None
        self._known_ids = set()
        self._formation = None          # latest {formation_class, confidence, from, to}
        self._last_formation_class = None
        self._min_formation_conf = 0.3  # below this, /formation is not treated as present
        self._dirty = False
        self._last_signature = None     # dedupe identical consecutive narrations
        self._model = None              # HotSwapClient, lazy; None until first A/C demand
        self._model_error = None        # remembered load failure (report once)

        self._pool = ThreadPoolExecutor(max_workers=1)

        self.create_subscription(TrackArray, '/tracks', self._on_tracks,
                                 qos_reliable(), callback_group=self._cbg)
        self.create_subscription(FingerprintSet, '/fingerprints', self._on_fp,
                                 qos_latched(), callback_group=self._cbg)
        self.create_subscription(FormationEstimate, '/formation', self._on_formation,
                                 qos_latched(), callback_group=self._cbg)
        # 2 Hz coalescing timer: acts only when a trigger is dirty.
        self.create_timer(0.5, self._process_if_dirty, callback_group=self._cbg)

        if self._keep_warm:
            self._pool.submit(self._ensure_model)  # opt-in warm start, off the executor

    # --- upstream callbacks: latch + detect triggers (cheap only) ---
    def _on_tracks(self, m):
        with self._lock:
            self._last_tracks = m
            ids = {t.global_id for t in m.tracks}
            if not ids <= self._known_ids:     # a new global_id appeared
                self._known_ids |= ids
                self._dirty = True

    def _on_fp(self, m):
        with self._lock:
            k = int(m.k_estimated)
            if k != self._last_k:              # k_estimated changed
                self._last_k = k
                self._dirty = True

    def _on_formation(self, m):
        # Real STGT /formation path is now live. Treat a formation as PRESENT once a
        # confident estimate arrives (below the floor => still degenerate/no-signal).
        # A change of formation_class is the (previously-disabled) formation trigger.
        with self._lock:
            if m.confidence < self._min_formation_conf:
                return
            prev = self._last_formation_class
            self._formation = {
                'formation_class': m.formation_class,
                'confidence': float(m.confidence),
                'from': prev,
                'to': m.formation_class,
            }
            if m.formation_class != prev:      # formation-change trigger (enabled)
                self._last_formation_class = m.formation_class
                self._dirty = True

    # --- coalesced trigger processing ---
    def _process_if_dirty(self):
        with self._lock:
            if not self._dirty or self._last_tracks is None:
                return
            self._dirty = False
            tracks = self._last_tracks
            k = self._last_k if self._last_k is not None else 0
            formation = dict(self._formation) if self._formation else None

        stamp = tracks.header.stamp
        t_s = stamp.sec + stamp.nanosec * 1e-9
        snapshot = {
            't_start_s': t_s, 't_end_s': t_s,
            'tracks': [{'global_id': t.global_id, 'status': int(t.status)}
                       for t in tracks.tracks],
            'k_estimated': k,
            'formation': formation,        # real STGT /formation when present, else None
        }

        ctx, key_windows, meta = build_ctx(snapshot)
        bucket_info = classify_ctx(ctx, key_windows)
        bucket = bucket_info['bucket']
        ref_ids = [t.global_id for t in tracks.tracks]

        if bucket == BUCKET_B:
            self._emit_abstention(bucket_info, meta, ref_ids, stamp)
        else:
            # Layer 1/3 — needs the model. Offload so the executor never blocks on GPU.
            self._pool.submit(self._run_model_path, ctx, key_windows, meta, ref_ids, stamp)

    # --- Layer 2 (v1 path): render + publish, torch-free, synchronous ---
    def _emit_abstention(self, bucket_info, meta, ref_ids, stamp):
        reasons = bucket_info.get('guard_reasons', [])
        signature = (meta['template_id'], tuple(reasons), meta.get('k_estimated'))
        if signature == self._last_signature:
            return                          # identical to the last one — don't spam
        self._last_signature = signature

        reason_txt = '; '.join(_GUARD_TEXT.get(r, r) for r in reasons) or 'guarded abstention'
        k = meta.get('k_estimated', 0)
        if meta.get('formation_present'):
            # a formation IS classified but this window routed to Layer 2 (e.g. below
            # the confidence threshold) — do NOT claim the classifier is offline.
            fc = meta.get('formation_class', 'unknown')
            conf = float(meta.get('formation_confidence', 0.0))
            text = (f'Formation classifier reports "{fc}" (confidence {conf:.2f}) across '
                    f'{k} tracked emitter(s), below the narration threshold. '
                    f'Holding — {reason_txt}.')
        else:
            text = (f'No formation signal yet: {k} emitter(s) tracked, formation classifier '
                    f'offline. Holding — {reason_txt}.')

        n = Narration()
        n.header.stamp = stamp
        n.header.frame_id = 'enu'
        n.text = text
        n.template_id = meta['template_id']          # "ctx_degenerate_no_formation"
        n.source = 'template'                         # machine-generated guard, no LLM
        n.severity = Narration.SEVERITY_INFO
        n.referenced_ids = ref_ids
        n.trigger_time = stamp
        self._pub.publish(n)
        self.health.bump()
        self.get_logger().info(
            f'narration [L2/{n.template_id}] guards={list(reasons)} -> "{text}"')

    # --- Layer 1/3 (DORMANT in v1): lazy GPU load + assess, runs on the 1-thread pool ---
    def _run_model_path(self, ctx, key_windows, meta, ref_ids, stamp):
        try:
            client, class_freq = self._ensure_model()
        except Exception as e:
            self.get_logger().error(f'model unavailable, cannot narrate A/C ctx: {e}')
            self.health.set_status(f'model load failed: {e}')
            return
        try:
            from demo_common import run_pipeline_case  # torch-heavy; lazy
            result = run_pipeline_case(client, class_freq, ctx, key_windows)
        except Exception as e:
            self.get_logger().error(f'assess_ctx failed: {e}')
            return
        if result.get('error'):
            self.get_logger().error('pipeline error: ' + result['error'])
            return
        self._publish_model_narration(result, meta, ref_ids, stamp)

    def _ensure_model(self):
        """Lazy-load the 4-bit base once, guarded by gpu_free_gib. Returns
        (client, class_freq). Raises on insufficient VRAM / missing CUDA."""
        with self._lock:
            if self._model is not None:
                return self._model
        # heavy imports isolated here — never reached on the v1 (Layer 2) path
        for extra in (os.path.join(_AADHYA, 'scripts'), os.path.join(_AADHYA, 'config')):
            if extra not in sys.path:
                sys.path.insert(0, extra)
        from demo_common import HotSwapClient, gpu_free_gib
        from swarm_intent.pipeline_v2 import default_class_freq
        import demo_config

        free = gpu_free_gib(self._required_gib)      # raises RuntimeError if too low
        self.get_logger().warn(f'loading narration model ({free:.1f} GiB free) — '
                               f'first A/C-routed ctx demanded it')
        client = HotSwapClient(demo_config.BASE_MODEL)
        client.add_adapter('active', str(demo_config.ACTIVE_ADAPTER))
        class_freq = default_class_freq()
        with self._lock:
            self._model = (client, class_freq)
        return self._model

    def _publish_model_narration(self, result, meta, ref_ids, stamp):
        a = result.get('assessment') or {}
        text = a.get('situation_summary') or a.get('threat_reasoning') or '(no summary)'
        threat = str(a.get('threat_level', 'unknown')).lower()
        severity = {'critical': Narration.SEVERITY_ALERT, 'high': Narration.SEVERITY_ALERT,
                    'medium': Narration.SEVERITY_NOTICE}.get(threat, Narration.SEVERITY_INFO)

        n = Narration()
        n.header.stamp = stamp
        n.header.frame_id = 'enu'
        n.text = text
        n.template_id = meta.get('template_id', f"layer_{result.get('layer')}")
        n.source = 'llm'                              # Layer 1 narrator / Layer 3 adapter
        n.severity = severity
        n.referenced_ids = ref_ids
        n.trigger_time = stamp
        self._pub.publish(n)
        self.health.bump()
        self.get_logger().info(
            f"narration [{result.get('layer')}] threat={threat} -> \"{text}\"")


def main():
    rclpy.init()
    node = NarrateNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._pool.shutdown(wait=False)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
