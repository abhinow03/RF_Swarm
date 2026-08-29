"""Generate review-ready PNGs for the demo2 Run A/B/C story, straight from real
captured RF IQ through the exact router+encoder code path the live pipeline
uses (rffp.router.ProtocolRouter + rffp.tiers.Tiers) -- no synthetic data, no
live pipeline required. The formation snapshot panel reuses the JSON already
captured by capture_run.py; it does not re-run anything.

Writes three PNGs to demo2/figures/:
  01_clustering.png       real embeddings (PCA) colored by true device, per run
  02_iq_waveform.png      one real captured I/Q burst per protocol
  03_formation_snapshot.png  final track positions vs the fixed receiver array
"""
import os, sys, json
import numpy as np
import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, '..', 'figures')
RUNS_DIR = os.path.join(HERE, '..', 'runs')
os.makedirs(FIG_DIR, exist_ok=True)

RFFP_ROOT = os.environ.get('RFFP_ROOT', os.path.expanduser('~/Desktop/RFFP_DEPLOY'))
DLMODEL_ROOT = os.environ.get('DLMODEL_ROOT', os.path.expanduser('~/CAPSTONE/DL_model'))
BLE_IQ_ROOT = os.environ.get('BLE_IQ_ROOT', '/home/docker/pw26_akp_01/ext_data/ble_xiao')
BLE_COLLECTIONS = ['Wireless (outdoors)/Loc1', 'Wireless (outdoors)/Loc2',
                   'Wireless (outdoors)/Loc3', 'Wireless (outdoors)/Loc4']

sys.path.insert(0, RFFP_ROOT)
sys.path.insert(0, DLMODEL_ROOT)
from rffp.router import ProtocolRouter, track_features
from rffp.tiers import Tiers

RUNS = [
    dict(name='Run A -- BLE (diamond)', run_id='runA_ble_nominal', protocol='ble',
         sample_rate=6.0e6, unit_ids=[4, 7, 28, 30, 18, 11]),
    dict(name='Run B -- WiFi (column)', run_id='runB_wifi_column', protocol='wifi',
         sample_rate=25.0e6, unit_ids=['1-1', '1-11', '12-7', '13-7', '10-4', '10-1']),
    dict(name='Run C -- Drone (v_shape)', run_id='runC_drone_vshape', protocol='drone_ocusync',
         sample_rate=50.0e6,
         unit_ids=['mavic3_1', 'mavic3S_1', 'mavicAir2s_2', 'mavicAir2s_6', 'mini3pro_1', 'mavicAir2s_3']),
]
WINDOWS_PER_UNIT = 60
rng = np.random.default_rng(0)


def load_ble_iq(unit_ids, per_unit):
    iq = {u: [] for u in unit_ids}
    for c in BLE_COLLECTIONS:
        d = os.path.join(BLE_IQ_ROOT, c)
        xp = os.path.join(d, 'X_train.npy')
        if not os.path.isfile(xp):
            continue
        X = np.load(xp, mmap_mode='r')
        Y = np.load(os.path.join(d, 'Y_train.npy'), mmap_mode='r')
        y = Y.argmax(1) if Y.ndim > 1 else Y
        take = max(1, per_unit // len(BLE_COLLECTIONS))
        for u in unit_ids:
            rows = np.where(y == u)[0]
            if len(rows) == 0:
                continue
            sel = rng.choice(rows, size=min(take, len(rows)), replace=False)
            iq[u].append(np.asarray(X[np.sort(sel)], np.float32))
    return {u: np.concatenate(v, axis=0)[:per_unit] for u, v in iq.items() if v}


def load_dlmodel_iq(loader_fn_name, unit_ids, per_unit):
    from gen_rff.data import loaders
    fn = getattr(loaders, loader_fn_name)
    iq = {}
    for u in unit_ids:
        units = fn([u], cap=per_unit)
        w = np.concatenate([np.asarray(v['X'], np.float32) for v in units], axis=0)
        iq[u] = w[:per_unit]
    return iq


def load_iq(protocol, unit_ids, per_unit):
    if protocol == 'ble':
        return load_ble_iq(unit_ids, per_unit)
    if protocol == 'wifi':
        return load_dlmodel_iq('load_wisig', unit_ids, per_unit)
    if protocol == 'drone_ocusync':
        return load_dlmodel_iq('load_drff_native', unit_ids, per_unit)
    raise ValueError(protocol)


print("loading router + tier encoders ...")
router = ProtocolRouter.load(os.path.join(RFFP_ROOT, 'router_state', 'router.pkl'))
tiers = Tiers()

# ---------- 1) clustering: real embeddings (PCA), colored by true device ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, run in zip(axes, RUNS):
    iq = load_iq(run['protocol'], run['unit_ids'], WINDOWS_PER_UNIT)
    all_emb, all_labels = [], []
    for u, w in iq.items():
        proto, conf, _ = router.route(track_features(w, run['sample_rate']))
        deep, _ = tiers.encode(proto, w)
        all_emb.append(np.asarray(deep))
        all_labels += [str(u)] * len(deep)
    X = np.concatenate(all_emb, axis=0)
    # t-SNE (cosine metric), not PCA: these embeddings are separated by *angle*
    # (the live base station clusters on a cosine-flavoured kNN graph, see
    # capstone_base/fingerprint.py) -- a linear PCA projection preserves global
    # variance, not that local neighbourhood structure, and visually smears
    # angularly-separated-but-similar-magnitude points together.
    perp = min(30, max(5, (len(X) - 1) // 3))
    X2 = TSNE(n_components=2, metric='cosine', perplexity=perp, random_state=0,
              init='random').fit_transform(X)
    uniq = sorted(set(all_labels))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(10, len(uniq))))
    for i, u in enumerate(uniq):
        m = [lbl == u for lbl in all_labels]
        ax.scatter(X2[m, 0], X2[m, 1], s=14, alpha=0.75, c=[cmap[i]], label=f"unit {u}")
    ax.set_title(run['name'])
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, loc='best')
fig.suptitle("Real-device embedding clustering -- each color is one real, distinct physical device\n"
              "(t-SNE of the same frozen-encoder output the live pipeline's K-estimation clusters on)")
fig.text(0.5, 0.01,
         "Honest caveat: BLE's real worst-pair cosine margin (0.761) is the tightest of the three protocols, "
         "so 4 of its 6 devices show partial overlap in this 2D projection even though the live 128-D "
         "clustering already resolves all 6 correctly at 0.81-0.99 confidence (see demo2/runs/runA_ble_nominal/"
         "04_clustering.json) -- a 2D-visualization limit, not a live-system failure.",
         ha='center', fontsize=7.5, style='italic', wrap=True)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig(os.path.join(FIG_DIR, '01_clustering.png'), dpi=130)
plt.close()
print("wrote 01_clustering.png")

# ---------- 2) raw I/Q waveform: one real captured burst per protocol ----------
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
for ax, run in zip(axes, RUNS):
    iq = load_iq(run['protocol'], run['unit_ids'][:1], 1)
    w = next(iter(iq.values()))[0]          # (2, n_samples): channel 0 = I, channel 1 = Q
    t = np.arange(w.shape[1]) / run['sample_rate'] * 1e6      # microseconds
    ax.plot(t, w[0], lw=0.8, label='I')
    ax.plot(t, w[1], lw=0.8, label='Q', alpha=0.8)
    ax.set_title(f"{run['name']} -- one real captured burst, unit {run['unit_ids'][0]}")
    ax.set_xlabel("time (us)"); ax.set_ylabel("amplitude")
    ax.legend(fontsize=8, loc='upper right')
fig.suptitle("Raw I/Q waveforms actually captured per protocol (not synthetic)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '02_iq_waveform.png'), dpi=130)
plt.close()
print("wrote 02_iq_waveform.png")

# ---------- 3) formation snapshot: from already-captured demo2 JSON, no rerun ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, run in zip(axes, RUNS):
    run_dir = os.path.join(RUNS_DIR, run['run_id'])
    with open(os.path.join(run_dir, '00_run_manifest.json')) as f:
        manifest = json.load(f)
    with open(os.path.join(run_dir, '06_localization.json')) as f:
        loc = json.load(f)
    with open(manifest['scenario_yaml_path']) as f:
        scen = yaml.safe_load(f)

    for i, r in enumerate(scen['receivers']):
        ax.scatter(r[0], r[1], marker='^', s=220, c='k', zorder=5)
        ax.annotate(f"RX{i}", (r[0], r[1]), textcoords='offset points', xytext=(6, 6), fontsize=8)
    xs = [t['position'][0] for t in loc['tracks']]
    ys = [t['position'][1] for t in loc['tracks']]
    ax.scatter(xs, ys, s=160, zorder=6, c='crimson')
    ax.set_title(f"{run['name']}\ntruth formation: {manifest['formation_label']}")
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    # zoom on the swarm itself (it's tens of metres across, the receiver array is
    # ~180 m across) so the formation shape is actually visible, not a speck
    cx, cy = np.mean(xs), np.mean(ys)
    half = max(25.0, max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 8)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
fig.suptitle("Final swarm snapshot from real EKF tracks -- one moment in time, zoomed on the swarm\n"
              "(receivers = fixed array ▲ where in frame, tracks = live position estimate ●)")
fig.text(0.5, 0.01,
         "Honest caveat: column (flat, ~25m span) reads clearly by eye. Diamond is really a ~20m-wide 3D "
         "octahedron (extends in altitude, not just East-North) and v_shape is a known-hard class -- both are "
         "close in scale to the documented real EKF position noise (~15-29m), so a single top-down snapshot "
         "doesn't visually read as a crisp shape even though the formation classifier uses the full 3D track "
         "history, not this 2D picture, to make its call.",
         ha='center', fontsize=7.5, style='italic', wrap=True)
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig(os.path.join(FIG_DIR, '03_formation_snapshot.png'), dpi=130)
plt.close()
print("wrote 03_formation_snapshot.png")
