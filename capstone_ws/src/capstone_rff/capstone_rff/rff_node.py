"""capstone_rff — fingerprint front-end: REAL RF IQ -> RFFP router + frozen encoder.

Runs the deployed RFFP pipeline (protocol router + frozen tier encoders,
RFFP_DEPLOY) on REAL IQ windows and publishes the resulting blind embedding on
/embeddings. Real receiver IQ flows through the DL protocol router and a frozen
tier encoder into the base station, exactly as the integrated-pipeline diagram
(section 2) specifies.

Per (burst, receiver): the emitter->unit binding (from /truth/association, sim-side
data-gen — the blind chain never sees it) selects a real unit; `windows_per_obs`
real IQ windows are drawn for that unit, routed by the RFFP ProtocolRouter, encoded
by the tier the ROUTER selects (no forced override — see _build_embedding_banks),
mean-pooled + re-L2'd, and published blind (burst_seq + rx_id only — NO unit label).

Protocol source (rf.protocol in the scenario YAML) selects the IQ ingestion path:
  ble             -> real ESP32-C3 BLE IQ (iq_root/collections .npy, unchanged path)
  wifi            -> real WiSig IQ (gen_rff.data.loaders.load_wisig, ~/CAPSTONE/DL_model)
  drone_ocusync   -> real DRFF airframe IQ (loaders.load_drff_native, same repo)
The wifi/drone paths need ~/CAPSTONE/DL_model on PYTHONPATH (activate.sh) + torch.
All three paths converge on the same router+tiers encode loop — the router's live
per-unit classification picks the tier encoder, not the scenario's declared protocol.

Validated equivalence (BLE only): encoding real IQ windows reproduces the old
embedding pool's per-unit centroids to cosine 0.999 (same frozen encoder). Needs
torch + sklearn (project .venv) + RFFP_DEPLOY on PYTHONPATH (activate.sh).
"""
import hashlib
import os

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from capstone_msgs.msg import Embedding, RfObservationSet, BurstAssociationTruth

from capstone_common.health import HealthReporter, qos_reliable

# RFFP deploy package (router + frozen tier encoders). On PYTHONPATH via activate.sh.
from rffp.router import ProtocolRouter, track_features
from rffp.tiers import Tiers, TIER_REGISTRY

DEFAULT_RFFP_ROOT = os.path.expanduser('~/Desktop/RFFP_DEPLOY')
DEFAULT_IQ_ROOT = '/home/docker/pw26_akp_01/ext_data/ble_xiao'
DEFAULT_COLLECTIONS = 'Wireless (outdoors)/Loc1,Wireless (outdoors)/Loc2,' \
                      'Wireless (outdoors)/Loc3,Wireless (outdoors)/Loc4'


class RffNode(Node):
    def __init__(self):
        super().__init__('capstone_rff')
        self.declare_parameter('passthrough_truth', False)
        self.declare_parameter('scenario', 'gate_k6')
        self.declare_parameter('scenario_path', '')
        self.declare_parameter('rffp_root', DEFAULT_RFFP_ROOT)
        self.declare_parameter('iq_root', DEFAULT_IQ_ROOT)
        self.declare_parameter('collections', DEFAULT_COLLECTIONS)
        self.declare_parameter('dlmodel_root', os.path.expanduser('~/CAPSTONE/DL_model'))
        self.declare_parameter('windows_per_unit', 800)     # per-unit IQ bank size (spans collections)
        pt = self.get_parameter('passthrough_truth').value

        scn = self._load_scenario()
        rff = scn['rff']
        self._k = int(scn['k'])
        self._protocol = scn['rf']['protocol']            # 'ble' | 'wifi' | 'drone_ocusync'
        self._sample_rate = float(scn['rf']['sample_rate_hz'])
        # unit_ids are ints for ble (Y-label indices into the .npy dataset), strings for
        # wifi (WiSig tx id, e.g. '1-1') / drone (DRFF airframe, e.g. 'mavic3_1')
        self._unit_ids = [int(u) for u in rff['unit_ids']] if self._protocol == 'ble' \
            else [str(u) for u in rff['unit_ids']]
        self._wpo = int(rff.get('windows_per_obs', 8))
        seed = int(scn.get('seed', 0))
        self._rng = np.random.default_rng([seed, 0x2FF])

        # --- load the RFFP deploy pipeline (router + frozen tier encoders) ---
        rffp_root = self.get_parameter('rffp_root').value
        router_pkl = os.path.join(rffp_root, 'router_state', 'router.pkl')
        if not os.path.exists(router_pkl):
            raise FileNotFoundError(
                f'RFFP router not found: {router_pkl} — fit it once:  '
                f'RFFP_REPO_ROOT=~/CAPSTONE/DL_model python {rffp_root}/fit_router.py')
        self._router = ProtocolRouter.load(router_pkl)
        self._tiers = Tiers()                                # frozen encoders lazy-load per protocol
        self._route_counts = {}

        # --- STARTUP: stream each unit's REAL IQ through the router + frozen encoder
        #     (on GPU) into per-unit fingerprint EMBEDDING banks. The encoding is
        #     amortized here (identical IQ need not be re-encoded every burst); per
        #     burst we then pool real embeddings from these banks. ---
        self._bank = self._build_embedding_banks()

        self._pub = self.create_publisher(Embedding, '/embeddings', qos_reliable())
        self.health = HealthReporter(self, pt)
        self._pending = {}
        self.create_subscription(RfObservationSet, '/rf_observations', self._on_obs, qos_reliable())
        self.create_subscription(BurstAssociationTruth, '/truth/association', self._on_assoc, qos_reliable())
        self.get_logger().info(
            f'rff: REAL-IQ({self._protocol}) -> RFFP router+encoder | router classes='
            f'{list(self._router.clf.classes_)} | units(emitter->unit)=' +
            ', '.join(f'{i}->{u}' for i, u in enumerate(self._unit_ids)))

    def _load_scenario(self):
        path = self.get_parameter('scenario_path').value
        if not path:
            share = get_package_share_directory('capstone_scenarios')
            path = os.path.join(share, 'scenarios', self.get_parameter('scenario').value + '.yaml')
        with open(path) as f:
            return yaml.safe_load(f)

    def _gather_iq_ble(self):
        iq_root = self.get_parameter('iq_root').value
        colls = [c.strip() for c in self.get_parameter('collections').value.split(',') if c.strip()]
        per_unit = int(self.get_parameter('windows_per_unit').value)
        # gather real IQ windows per unit across the collections (spans channels)
        iq = {u: [] for u in self._unit_ids}
        for c in colls:
            d = os.path.join(iq_root, c)
            xp = os.path.join(d, 'X_train.npy')
            if not os.path.isfile(xp):
                continue
            X = np.load(xp, mmap_mode='r')
            Y = np.asarray(np.load(os.path.join(d, 'Y_train.npy'), mmap_mode='r'))
            y = Y.argmax(1) if Y.ndim > 1 else Y
            take = max(1, per_unit // len(colls))
            for u in self._unit_ids:
                rows = np.where(y == u)[0]
                if len(rows) == 0:
                    continue
                sel = self._rng.choice(rows, size=min(take, len(rows)), replace=False)
                iq[u].append(np.asarray(X[np.sort(sel)], np.float32))
        for u in self._unit_ids:
            if not iq[u]:
                raise ValueError(f'no BLE IQ found for unit {u} under {iq_root} / {colls}')
        return {u: np.concatenate(v, axis=0) for u, v in iq.items()}

    def _gather_iq_dlmodel(self, loader_fn, kind):
        """wifi (load_wisig) / drone_ocusync (load_drff_native) real IQ, via the same
        ~/CAPSTONE/DL_model repo the router itself was fit from (RFFP_REPO_ROOT in
        fit_router.py) — real held-out data, not synthetic demo-only content. Each
        unit is loaded individually (matching gen_rff.demo_router.replay's own
        per-device pooling pattern) and windows are capped/concatenated the same way
        the BLE path caps per-collection."""
        import sys
        dl_root = self.get_parameter('dlmodel_root').value
        if dl_root not in sys.path:
            sys.path.insert(0, dl_root)
        from gen_rff.data import loaders
        per_unit = int(self.get_parameter('windows_per_unit').value)
        fn = getattr(loaders, loader_fn)
        iq = {}
        for u in self._unit_ids:
            units = fn([u], cap=per_unit)
            if not units:
                raise ValueError(f'no real {kind} IQ found for unit {u!r} under {dl_root} '
                                  f'(gen_rff.data.loaders.{loader_fn})')
            w = np.concatenate([np.asarray(v['X'], np.float32) for v in units], axis=0)
            if len(w) > per_unit:
                sel = self._rng.choice(len(w), size=per_unit, replace=False)
                w = w[np.sort(sel)]
            iq[u] = w
        return iq

    def _build_embedding_banks(self):
        import time
        t0 = time.time()
        # 1) gather real IQ windows per unit — path depends on the scenario's declared
        #    protocol (ground truth for WHICH DATASET to pull from; the router below still
        #    independently classifies each unit blind, and its decision — not this — picks
        #    the encoder tier).
        if self._protocol == 'ble':
            iq = self._gather_iq_ble()
        elif self._protocol == 'wifi':
            iq = self._gather_iq_dlmodel('load_wisig', 'wifi')
        elif self._protocol == 'drone_ocusync':
            iq = self._gather_iq_dlmodel('load_drff_native', 'drone_ocusync')
        else:
            raise ValueError(f'unknown rf.protocol {self._protocol!r} — expected ble|wifi|drone_ocusync')
        # 2) route + encode each unit's real IQ -> per-window embedding bank. The router's
        #    live decision selects the tier encoder directly (no forced override) — this is
        #    the thing that makes the router non-decorative: a genuinely different-protocol
        #    unit gets a genuinely different (dim, checkpoint) encoder.
        out = {}
        self._route_dims = {}
        self._route_protos = {}
        for u in self._unit_ids:
            w = iq[u]                                         # (n, 2, window_samples) real IQ
            # protocol router on the whole unit (rate-aware features + OOD gate)
            proto, conf, _ = self._router.route(track_features(w, self._sample_rate))
            self._route_counts[proto] = self._route_counts.get(proto, 0) + 1
            self._route_protos[u] = proto
            deep, _aux = self._tiers.encode(proto, w)          # (n, dim) L2-normed, frozen tier encoder (GPU)
            out[u] = np.ascontiguousarray(deep, np.float32)
            self._route_dims[u] = out[u].shape[1]
        dims = set(self._route_dims.values())
        if len(dims) > 1:
            raise ValueError(
                f'mixed embedding dims within one run — router split units across tiers: '
                f'{self._route_dims} (route_counts={self._route_counts}). This scenario is not '
                f'single-protocol as assumed; downstream clustering cannot pool mixed dims.')
        # single tier used across this run (dims uniform, just asserted) -> sha256 the actual
        # checkpoint file the router selected, real provenance rather than a hardcoded constant
        self._route_label = next(iter(self._route_protos.values()))
        ckpt = TIER_REGISTRY[self._route_label]['ckpt']
        with open(ckpt, 'rb') as f:
            self._encoder_sha = hashlib.sha256(f.read()).hexdigest()
        # 3) free the GPU encoder — runtime only pools from the numpy banks
        try:
            import torch
            del self._tiers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.get_logger().info(
            f'rff: pre-encoded real IQ -> embedding banks in {time.time()-t0:.1f}s | '
            f'router routes={self._route_counts} | '
            + ', '.join(f'u{u}:{len(out[u])}emb' for u in self._unit_ids))
        return out

    # --- correlate obs (burst -> rx list) with association (burst -> emitter) ---
    def _on_obs(self, m):
        e = self._pending.setdefault(m.burst_seq, {})
        e['rx'] = [o.rx_id for o in m.observations]
        e['stamp'] = m.header.stamp
        self._maybe_emit(m.burst_seq)

    def _on_assoc(self, m):
        e = self._pending.setdefault(m.burst_seq, {})
        e['emitter'] = int(m.truth_emitter_id)
        self._maybe_emit(m.burst_seq)

    def _maybe_emit(self, bseq):
        e = self._pending.get(bseq)
        if not e or 'rx' not in e or 'emitter' not in e:
            return
        del self._pending[bseq]
        unit = self._unit_ids[e['emitter']]
        bank = self._bank[unit]                               # pre-encoded real-IQ embeddings (n, 128)
        for rx_id in e['rx']:
            idx = self._rng.choice(len(bank), size=min(self._wpo, len(bank)), replace=False)
            v = bank[idx].mean(0)                             # pool windows_per_obs real embeddings
            v = v / (np.linalg.norm(v) + 1e-8)               # re-L2 after pooling
            msg = Embedding()
            msg.header.stamp = e['stamp']
            msg.header.frame_id = 'enu'
            msg.burst_seq = bseq
            msg.rx_id = int(rx_id)
            msg.dim = len(v)
            msg.vector = v.astype(np.float32).tolist()
            msg.encoder_sha256 = self._encoder_sha
            msg.source = f'rffp:router+{self._route_label}_encoder'
            self._pub.publish(msg)
        self.health.bump()


def main():
    rclpy.init()
    node = RffNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
