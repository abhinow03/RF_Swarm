#!/usr/bin/env python3
"""capture_run.py — subscribes to the REAL pipeline's published topics only (no synthetic
data path) and writes one coherent run-artifact under demo2/runs/<run_id>/. Every JSON field
is sourced from a message a real node actually published — this script computes nothing the
pipeline didn't already compute, except two explicitly-labeled truth-informed checks (07/11)
that mirror capstone_eval's existing truth-privileged role for scoring, not pipeline behavior.

Usage (run AFTER `run_pipeline.sh <scenario> sim base track formation predict bridge` and the
matching rff/narrate nodes are up and settled):
    python3 capture_run.py --scenario gate_k6 --run-id runA_ble_encirclement --duration 60
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import Log
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

from capstone_msgs.msg import (Embedding, FingerprintSet, BurstAssociationSet, TrackArray,
                               FormationEstimate, PathPrediction, Narration, NodeHealth,
                               RfObservationSet, BurstAssociationTruth, TruthScene)

REPO_ROOT = os.path.expanduser('~/RF_Swarm')
DEMO2_ROOT = os.path.join(REPO_ROOT, 'demo2')
STATUS_NAME = {0: 'init', 1: 'tracking', 2: 'coasting', 3: 'lost'}


def qos_reliable(depth=10):
    return QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=depth)


def qos_latched(depth=1):
    return QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=depth)


def qos_best_effort(depth=10):
    return QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=depth)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def stamp_to_float(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class RunCapture(Node):
    def __init__(self, scenario, scenario_path):
        super().__init__('demo2_capture')
        self.scenario = scenario
        self.scenario_path = scenario_path

        self.embeddings = []            # list of (dim, encoder_sha256, source, burst_seq, rx_id)
        self.rff_router_log = None      # parsed from /rosout (rff's own startup INFO line)
        self.fingerprints = None        # latest FingerprintSet
        self.associations = None        # latest BurstAssociationSet
        self.tracks = None              # latest TrackArray
        self.formation = None           # latest FormationEstimate
        self.formation_history = []     # every FormationEstimate this window (stability check)
        self.predictions = {}           # global_id -> latest PathPrediction
        self.narrations = []            # ALL narration events (sparse, keep full history)
        self.health = {}                # node_name -> latest NodeHealth
        self.rf_observations = []       # summary stats accumulator (not full history)
        self.truth_association = {}     # burst_seq -> truth_emitter_id (truth-informed check ONLY)
        self.truth_scene = None         # latest TruthScene (truth-informed check ONLY)

        self.create_subscription(Log, '/rosout', self._on_rosout, qos_best_effort())
        self.create_subscription(Embedding, '/embeddings', self._on_embedding, qos_reliable())
        self.create_subscription(FingerprintSet, '/fingerprints', self._on_fp, qos_latched())
        self.create_subscription(BurstAssociationSet, '/associations', self._on_assoc, qos_latched())
        self.create_subscription(TrackArray, '/tracks', self._on_tracks, qos_reliable())
        self.create_subscription(FormationEstimate, '/formation', self._on_formation, qos_latched())
        self.create_subscription(PathPrediction, '/predictions', self._on_pred, qos_reliable())
        self.create_subscription(Narration, '/narration', self._on_narration, qos_latched())
        self.create_subscription(RfObservationSet, '/rf_observations', self._on_obs, qos_reliable())
        self.create_subscription(BurstAssociationTruth, '/truth/association', self._on_truth_assoc, qos_reliable())
        self.create_subscription(TruthScene, '/truth/scene', self._on_truth_scene, qos_best_effort())
        for node_name in ['capstone_sim', 'capstone_rff', 'capstone_base', 'capstone_track',
                          'capstone_formation', 'capstone_predict', 'capstone_narrate',
                          'capstone_bridge']:
            self.create_subscription(NodeHealth, f'/health/{node_name}', self._on_health,
                                     qos_best_effort())

    def _on_rosout(self, m):
        if m.name != 'capstone_rff':
            return
        mo = re.search(r"router routes=(\{[^}]*\})", m.msg)
        if mo:
            try:
                counts = eval(mo.group(1), {'__builtins__': {}})
            except Exception:
                counts = None
            self.rff_router_log = dict(raw_line=m.msg, route_counts=counts)

    def _on_embedding(self, m):
        self.embeddings.append(dict(dim=int(m.dim), encoder_sha256=m.encoder_sha256,
                                    source=m.source, burst_seq=int(m.burst_seq), rx_id=int(m.rx_id)))

    def _on_fp(self, m):
        self.fingerprints = m

    def _on_assoc(self, m):
        self.associations = m

    def _on_tracks(self, m):
        self.tracks = m

    def _on_formation(self, m):
        self.formation = m
        self.formation_history.append(dict(formation_class=m.formation_class,
                                            confidence=m.confidence, from_truth=m.from_truth))

    def _on_pred(self, m):
        self.predictions[m.global_id] = m

    def _on_narration(self, m):
        self.narrations.append(dict(
            text=m.text, template_id=m.template_id, source=m.source, severity=int(m.severity),
            referenced_ids=list(m.referenced_ids), trigger_time=stamp_to_float(m.trigger_time)))

    def _on_obs(self, m):
        for o in m.observations:
            self.rf_observations.append(dict(
                toa_sigma=o.toa_sigma, rssi_dbm=o.rssi_dbm, rssi_sigma=o.rssi_sigma,
                fdoa_hz=o.fdoa_hz, fdoa_sigma=o.fdoa_sigma, snr_db=o.snr_db,
                nominal_freq_hz=o.nominal_freq_hz))

    def _on_truth_assoc(self, m):
        self.truth_association[int(m.burst_seq)] = int(m.truth_emitter_id)

    def _on_truth_scene(self, m):
        self.truth_scene = m

    def _on_health(self, m):
        self.health[m.node_name] = dict(
            passthrough_truth=bool(m.passthrough_truth), publish_rate_hz=m.publish_rate_hz,
            messages_dropped=int(m.messages_dropped), status_text=m.status_text)


def summarize(values):
    if not values:
        return None
    import statistics
    return dict(n=len(values), mean=statistics.fmean(values),
               stdev=statistics.pstdev(values) if len(values) > 1 else 0.0,
               min=min(values), max=max(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--duration', type=float, default=60.0)
    args = ap.parse_args()

    share = get_package_share_directory('capstone_scenarios')
    scenario_path = os.path.join(share, 'scenarios', args.scenario + '.yaml')
    src_scenario_path = os.path.join(REPO_ROOT, 'capstone_ws', 'src', 'capstone_scenarios',
                                     'scenarios', args.scenario + '.yaml')

    import yaml
    with open(scenario_path) as f:
        scn = yaml.safe_load(f)

    rclpy.init()
    node = RunCapture(args.scenario, scenario_path)
    t_start = time.time()
    while time.time() - t_start < args.duration:
        rclpy.spin_once(node, timeout_sec=0.5)
    t_end = time.time()

    run_dir = os.path.join(DEMO2_ROOT, 'runs', args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    def write(name, obj):
        with open(os.path.join(run_dir, name), 'w') as f:
            json.dump(obj, f, indent=2, default=str)

    # 00_run_manifest.json — provenance. No git available anywhere in RF_Swarm (verified);
    # vendor/model provenance sourced from DEPENDENCY_MANIFEST.md instead.
    dep_manifest_path = os.path.join(REPO_ROOT, 'DEPENDENCY_MANIFEST.md')
    write('00_run_manifest.json', dict(
        run_id=args.run_id, scenario_id=scn['scenario_id'], protocol=scn['rf']['protocol'],
        formation_label=scn['formation_label'], k=scn['k'], seed=scn.get('seed'),
        capture_start_unix=t_start, capture_end_unix=t_end, duration_s=t_end - t_start,
        scenario_yaml_path=src_scenario_path,
        scenario_yaml_sha256=sha256_file(src_scenario_path) if os.path.exists(src_scenario_path) else None,
        dependency_manifest_sha256=sha256_file(dep_manifest_path) if os.path.exists(dep_manifest_path) else None,
        node_health=node.health,
        note='No git repo exists anywhere under RF_Swarm (verified 2026-08-28) — code provenance '
             'is the scenario YAML sha256 above + DEPENDENCY_MANIFEST.md (vendor commit, model '
             'checksums), not a git commit hash.'))

    # 01_input_rf.json — scenario config (not a measurement) + real observation volume
    write('01_input_rf.json', dict(
        protocol=scn['rf']['protocol'], unit_ids=scn['rff']['unit_ids'],
        rf_physics=scn['rf'], n_observations_captured=len(node.rf_observations)))

    # 02_router.json — per NOISE_MODEL/task decision: report what actually happens today
    # (router classifies once per unit at rff startup, result only ever logged, never
    # published on a topic) rather than inventing new per-burst node behavior.
    write('02_router.json', dict(
        granularity='per-unit, at rff_node startup (pre-encode) — NOT per-burst; the router '
                    'does not re-classify at runtime, matching the current node design',
        source='parsed from /rosout (capstone_rff\'s own startup log line) — zero new pipeline behavior',
        **(node.rff_router_log or {'raw_line': None, 'route_counts': None,
                                    'note': 'no capstone_rff startup line seen during capture window '
                                            '— rff_node must be started before capture begins'})))

    # 03_embeddings.json — metadata + summary only, NOT full vectors
    dims = {e['dim'] for e in node.embeddings}
    shas = {e['encoder_sha256'] for e in node.embeddings}
    sources = {e['source'] for e in node.embeddings}
    write('03_embeddings.json', dict(
        n_messages=len(node.embeddings), dims=sorted(dims), encoder_sha256=sorted(shas),
        sources=sorted(sources), rx_ids_seen=sorted({e['rx_id'] for e in node.embeddings}),
        burst_seq_range=[min((e['burst_seq'] for e in node.embeddings), default=None),
                        max((e['burst_seq'] for e in node.embeddings), default=None)]))

    # 04_clustering.json — latest FingerprintSet (base station), full assignment list
    fp = node.fingerprints
    write('04_clustering.json', None if fp is None else dict(
        k_estimated=fp.k_estimated, eigenvalues=list(fp.eigenvalues),
        eigengap_value=fp.eigengap_value, n_accumulated=fp.n_accumulated,
        encoder_sha256=fp.encoder_sha256,
        assignments=[dict(burst_seq=int(a.burst_seq), global_id=a.global_id,
                          confidence=a.confidence) for a in fp.assignments]))

    # 05_observables.json — TDoA/FDoA/RSSI summary stats (pre-registered sigma is in NOISE_MODEL.md)
    write('05_observables.json', dict(
        toa_sigma=summarize([o['toa_sigma'] for o in node.rf_observations]),
        rssi_dbm=summarize([o['rssi_dbm'] for o in node.rf_observations]),
        fdoa_hz=summarize([o['fdoa_hz'] for o in node.rf_observations]),
        snr_db=summarize([o['snr_db'] for o in node.rf_observations]),
        pre_registered_physics_ref='capstone_ws/src/capstone_scenarios/NOISE_MODEL.md'))

    # 06_localization.json — latest TrackArray
    tr = node.tracks
    write('06_localization.json', None if tr is None else dict(
        n_tracks=len(tr.tracks),
        tracks=[dict(global_id=t.global_id,
                     position=[t.position.x, t.position.y, t.position.z],
                     velocity=[t.velocity.x, t.velocity.y, t.velocity.z],
                     covariance_diag=[t.covariance[i] for i in (0, 7, 14, 21, 28, 35)],
                     gdop=t.gdop, n_receivers_used=int(t.n_receivers_used),
                     status=STATUS_NAME.get(t.status, t.status))
                for t in tr.tracks]))

    # 07_association.json — burst->global_id map + a TRUTH-INFORMED bijectivity check
    # (mirrors capstone_eval's existing truth-privileged scoring role; the pipeline itself
    # never sees this comparison, it is verification evidence only)
    assoc = node.associations
    bij = None
    by_global = {}
    if assoc is not None and node.truth_association:
        by_global = defaultdict(set)
        by_truth = defaultdict(set)
        for a in assoc.associations:
            bseq = int(a.burst_seq)
            if bseq in node.truth_association:
                by_global[a.global_id].add(node.truth_association[bseq])
                by_truth[node.truth_association[bseq]].add(a.global_id)
        bij = dict(
            note='TRUTH-INFORMED verification only (like capstone_eval) — not used by the pipeline',
            global_id_to_n_distinct_truth_emitters={k: len(v) for k, v in by_global.items()},
            truth_emitter_to_n_distinct_global_ids={k: len(v) for k, v in by_truth.items()},
            is_bijective=all(len(v) == 1 for v in by_global.values()) and
                        all(len(v) == 1 for v in by_truth.values()))
    write('07_association.json', None if assoc is None else dict(
        k_estimated=assoc.k_estimated, encoder_sha256=assoc.encoder_sha256,
        n_associations=len(assoc.associations),
        associations=[dict(burst_seq=int(a.burst_seq), global_id=a.global_id,
                           confidence=a.confidence) for a in assoc.associations],
        bijectivity_check=bij))

    # 08_formation.json — latest snapshot AND a stability view over the whole capture window.
    # A single latest message can land mid-flicker (same issue fixed client-side in the AR
    # app with majority-vote debounce on /formation) — report both, honestly, rather than
    # picking whichever message happened to arrive last.
    fm = node.formation
    from collections import Counter
    label_counts = Counter(h['formation_class'] for h in node.formation_history)
    majority_label, majority_n = (label_counts.most_common(1)[0] if label_counts else (None, 0))
    write('08_formation.json', None if fm is None else dict(
        latest=dict(formation_class=fm.formation_class, confidence=fm.confidence,
                    class_probabilities=list(fm.class_probabilities), member_ids=list(fm.member_ids),
                    from_truth=fm.from_truth),
        stability=dict(n_messages=len(node.formation_history),
                       label_counts=dict(label_counts),
                       majority_label=majority_label,
                       majority_fraction=(majority_n / len(node.formation_history)
                                         if node.formation_history else None),
                       is_stable=(majority_n == len(node.formation_history)) if node.formation_history else None),
        warning=None if not fm.from_truth else
               'from_truth=True — this run classified PARAMETRIC TRUTH, not the real EKF '
               'tracks (debug/isolation mode). Not representative of the real pipeline.'))

    # 09_prediction.json
    write('09_prediction.json', {
        gid: dict(horizon_sec=p.horizon_sec, model_id=p.model_id,
                 n_waypoints=len(p.waypoints),
                 waypoints=[[w.x, w.y, w.z] for w in p.waypoints],
                 position_sigma=list(p.position_sigma))
        for gid, p in node.predictions.items()})

    # 10_narration.json — full event history (sparse, meaningful individually)
    write('10_narration.json', dict(n_events=len(node.narrations), events=node.narrations))

    # 11_run_summary.json — headline numbers + pass/fail vs pre-registered gates.
    # Position RMSE is a TRUTH-INFORMED check (same precedent as 07's bijectivity).
    rmse = None
    if tr is not None and node.truth_scene is not None:
        truth_by_id = {int(e.emitter_id): e for e in node.truth_scene.emitters}
        # global_id "d1".."dK" -> truth index via the bijective map from 07, if available
        errs = []
        if bij and bij['is_bijective']:
            g2t = {g: next(iter(v)) for g, v in by_global.items()}
            for t in tr.tracks:
                tid = g2t.get(t.global_id)
                if tid is not None and tid in truth_by_id:
                    e = truth_by_id[tid]
                    dx = t.position.x - e.position.x
                    dy = t.position.y - e.position.y
                    dz = t.position.z - e.position.z
                    errs.append((dx * dx + dy * dy + dz * dz) ** 0.5)
        if errs:
            import statistics
            rmse = dict(note='TRUTH-INFORMED verification only, not used by the pipeline',
                       n=len(errs), rmse_m=(statistics.fmean(e * e for e in errs)) ** 0.5,
                       mean_m=statistics.fmean(errs), max_m=max(errs))
    write('11_run_summary.json', dict(
        run_id=args.run_id, scenario_id=scn['scenario_id'], protocol=scn['rf']['protocol'],
        formation_label_truth=scn['formation_label'],
        k_true=scn['k'], k_estimated=(fp.k_estimated if fp else None),
        k_estimate_correct=(fp.k_estimated == scn['k']) if fp else None,
        formation_class_latest=(fm.formation_class if fm else None),
        formation_class_majority=majority_label,
        formation_majority_fraction=(majority_n / len(node.formation_history)
                                     if node.formation_history else None),
        formation_confidence=(fm.confidence if fm else None),
        formation_correct=(majority_label == scn['formation_label']) if majority_label else None,
        n_tracks=(len(tr.tracks) if tr else None),
        position_rmse=rmse,
        n_narration_events=len(node.narrations),
        node_health_status={k: v['status_text'] for k, v in node.health.items()}))

    node.destroy_node()
    rclpy.shutdown()
    print(f'wrote {run_dir}')


if __name__ == '__main__':
    main()
