# RF_Swarm

Real-time RF-fingerprint identification, TDoA/FDOA localization, and formation
classification for a swarm of emitters — fed by **real captured RF signals**
(BLE, WiFi, and drone/OcuSync), running as a ROS 2 pipeline, visualized live in
AR on a Meta Quest headset.

This README is the map. If you're new to the project, read this top to bottom
once — every deeper doc it links to assumes you've read this first.

---

## Table of contents

1. [What this project actually does](#what-this-project-actually-does)
2. [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
3. [Repository map](#repository-map)
4. [The two environments (and why)](#the-two-environments-and-why)
5. [Running it](#running-it)
6. [Scenarios](#scenarios)
7. [The `demo2` review pipeline](#the-demo2-review-pipeline)
8. [The AR headset client (separate repo)](#the-ar-headset-client-separate-repo)
9. [Project history](#project-history)
10. [Known gaps — read this before demoing](#known-gaps--read-this-before-demoing)
11. [Doc index](#doc-index)

---

## What this project actually does

A swarm of `K` RF emitters (BLE beacons, WiFi devices, or drones) moves in a
rigid formation (v-shape, diamond, column, encirclement, ...). A fixed array
of 5 receivers picks up their bursts. Nothing about which emitter is which,
how many there are, or what formation they're in is given to the system —
all of it is estimated from the RF signal alone:

1. **Identify emitters** from their RF fingerprint (hardware-level signal
   quirks — CFO, IQ imbalance — captured by a neural embedding), without ever
   being told how many emitters exist or what their IDs are.
2. **Localize** each emitter in 3D from time-difference-of-arrival (TDoA)
   across the receiver array, while simultaneously estimating each receiver's
   own clock bias/drift (the array's clocks are not synchronized by hardware).
3. **Classify the formation shape** the swarm is flying in, from a rolling
   window of position tracks, using a graph transformer.
4. **Predict** where the swarm is headed.
5. **Narrate** what's happening in plain language (rule-based templates, with
   an LLM tier for richer context when GPU budget allows).
6. **Visualize** all of the above live, in AR, on a headset.

The whole thing runs as a set of ROS 2 nodes, each doing exactly one of these
jobs, talking to each other only over ROS topics — no node is given
information it wouldn't have in a real deployment (ground truth is only ever
used for scoring, never as an input to any estimator).

## The pipeline, stage by stage

```
 real RF IQ samples (BLE / WiFi / drone_ocusync)
        |
        v
 capstone_sim -------------------------> /truth/scene, /truth/receivers   (ground truth; NOT consumed by estimators)
        |                                /rf_observations, /receivers
        v
 capstone_rff        (protocol router + frozen tier encoder -> embedding)
        |
        v  /embeddings
 capstone_base        (blind: eigengap K-estimation + spectral clustering)
        |
        v  /fingerprints, /associations
 capstone_track        (TDoA init + iterated EKF, joint clock-bias estimation)
        |
        v  /tracks
        +------------------------+------------------------+
        v                        v                        v
 capstone_formation      capstone_predict          capstone_narrate
 (STGT graph transformer) (constant-velocity        (template + LLM
        |                  baseline predictor)        narration)
        v  /formation             v  /predictions             v  /narration
        +------------------------+------------------------+
                                  v
                          capstone_bridge
                                  |
                                  v  /scene_update (JSON over rosbridge/wss)
                          AR headset client (separate repo, see below)
```

`capstone_eval` sits off to the side — it's the *only* node allowed to read
`/truth/*`, and only for scoring estimator output against ground truth
(RMSE, NEES), never for feeding an estimator.

| Package | Job | Publishes |
|---|---|---|
| `capstone_msgs` | Frozen `.msg` contract (19 types). No code. | — |
| `capstone_scenarios` | Pure-YAML scenario configs + the noise-model derivation. No code. | — |
| `capstone_sim` | Time authority + scripted ground-truth kinematics for a scenario. | `/clock`, `/truth/scene`, `/truth/receivers`, `/receivers`, `/truth/association`, `/rf_observations` |
| `capstone_rff` | Routes real RF IQ through the protocol classifier + frozen tier encoder, publishes blind (unlabeled) embeddings. | `/embeddings` |
| `capstone_base` | Blind base station: eigengap K-estimation + spectral partition, assigns global IDs. | `/fingerprints`, `/associations` |
| `capstone_track` | TDoA closed-form init + joint iterated EKF (position + per-receiver clock nuisance states). | `/tracks` |
| `capstone_formation` | STGT graph-transformer formation classifier over a rolling track window. | `/formation` |
| `capstone_predict` | Constant-velocity baseline path predictor. | `/predictions` |
| `capstone_narrate` | Tactical narration: template tier (always live) + LLM tier (GPU, lazy-loaded). | `/narration` |
| `capstone_bridge` | Flattens `/tracks` (+confidence from `/fingerprints`) into the AR client's JSON wire format, ENU→Y-up conversion. | `/scene_update` |
| `capstone_eval` | Truth-privileged scoring only (RMSE/NEES vs. `/truth/*`). | `/health/capstone_eval` |
| `capstone_common` | Shared library: QoS profile factories, `NodeHealth` publisher. Not a node. | `/health/<node>` |

Full message-field-level contract: [`capstone_msgs_spec.md`](capstone_msgs_spec.md).
Full wire-format decision record for `/scene_update`: [`reconciliation_answers.md`](reconciliation_answers.md).

## Repository map

```
RF_Swarm/
├── capstone_ws/                 ROS 2 Jazzy workspace (the actual pipeline)
│   ├── activate.sh              source this first, every time
│   ├── scripts/                 launch scripts (see "Running it" below)
│   ├── src/                     one ROS 2 package per pipeline stage (table above)
│   ├── client/                  drop-in JS bridge helpers used by the AR client
│   └── certs/                   TLS certs for the AR headset link (gitignored — generate locally)
├── vendor/                      pinned, checksummed peer code (see DEPENDENCY_MANIFEST.md)
│   └── src/swarm_intent/        the STGT formation model + narration context builder
├── models/
│   ├── stgt/                    STGT classifier weights (val_acc 0.868)
│   └── llm_adapter/             LoRA adapter for the narration LLM (git-lfs)
├── data/                        real BLE embedding pool (Phase 3 provenance)
├── demo/                        earlier scripted-demo capture (superseded by demo2)
├── demo2/                       review-ready run instrumentation — see below
├── capstone_msgs_spec.md        the message contract, field by field
├── DEPENDENCY_MANIFEST.md       provenance ledger for every vendored/copied artifact
├── README_INTEGRATED_SYSTEM.md  earlier system-integration notes (launch commands, directory tree)
├── reconciliation_answers.md    the /scene_update wire-format decision record
└── requirements.txt             pip freeze of the CPU venv (see "environments" below)
```

## The two environments (and why)

There are **two** Python environments involved, and this is deliberate, not
accidental:

1. **Project `.venv`** (this repo, `requirements.txt`) — CPU-only. Runs every
   node except the two GPU-heavy code paths below. Built with
   `--system-site-packages` on top of a system ROS 2 Jazzy install (so
   `ros2 run`'s system-Python shebang still sees `rclpy` while also getting
   this venv's `sklearn`/`scipy`/CPU `torch`/`torch_geometric`).
2. **A separate CUDA venv**, not part of this repo (`~/aadhya/uav-llm/.venv`
   on the machine this was built on) — used *only* by two launch scripts:
   - `capstone_rff`'s real-IQ pre-encoding (GPU makes this ~30x faster than
     CPU at startup, then the GPU is freed).
   - `capstone_narrate`'s LLM tier (4-bit Qwen2.5-7B + LoRA adapter).

   The **code** these two scripts run is fully vendored into this repo
   (`vendor/`, `models/llm_adapter/`) — only the heavyweight CUDA *interpreter
   and its installed packages* (torch+CUDA, transformers, bitsandbytes) live
   outside the repo, because duplicating a multi-GB CUDA stack per-repo isn't
   practical. If you're setting this up fresh on a machine with a GPU, you
   need a second venv with `torch` (CUDA build), `transformers`,
   `bitsandbytes`, `peft` installed — nothing else is special about it.

   One additional dependency **not vendored**: the Qwen2.5-7B base model
   itself (~15GB, lives in the HF cache on the original machine). See
   [`DEPENDENCY_MANIFEST.md`](DEPENDENCY_MANIFEST.md) §5 for exact commands to
   fully localize it if you need to.

Two more environment variables `activate.sh` needs to know about for the
non-BLE scenarios (Run B/C in `demo2/`, see below):
- `RFFP_ROOT` — the protocol router + frozen tier encoders. Defaults to
  `~/Desktop/RFFP_DEPLOY`.
- `DLMODEL_ROOT` — real WiFi/drone IQ loaders, only needed if you run a
  `wifi`/`drone_ocusync` scenario. Defaults to `~/CAPSTONE/DL_model`.

Neither of these is vendored into this repo (they're large external
datasets/tools, not code this project owns) — if you don't have them, every
`ble` scenario still works fully; only `wifi`/`drone_ocusync` scenarios need
them.

## Running it

```bash
# 1. one-time: create the CPU venv and install deps
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate

# 2. every session: source the environment (ROS + venv + vendor + PYTHONPATH)
source capstone_ws/activate.sh

# 3. build the workspace once (or after any capstone_msgs/.py change)
cd capstone_ws && colcon build --symlink-install && cd ..
source capstone_ws/install/setup.bash

# 4. launch the CPU-only nodes for a scenario
bash capstone_ws/scripts/run_pipeline.sh gate_k6 sim base track formation predict bridge

# 5. in a second terminal (same activate.sh sourced): launch the GPU stages
#    -- these two need the CUDA venv, see above
bash capstone_ws/scripts/run_rff_real.sh gate_k6
bash capstone_ws/scripts/run_narrate_llm.sh

# 6. optional: expose rosbridge over wss:// for the AR headset
bash capstone_ws/scripts/run_rosbridge_wss.sh
```

`run_pipeline.sh` cleans up any leftover `capstone_*` node processes both
before it starts and on exit — if a previous run died uncleanly, just re-run
it, don't hand-kill things unless you know what you're doing (`ps -o
pid,user,cmd` first, always).

## Scenarios

Every scenario is one YAML file under
`capstone_ws/src/capstone_scenarios/scenarios/`. Physics are pre-registered
*before* results exist — see
[`NOISE_MODEL.md`](capstone_ws/src/capstone_scenarios/NOISE_MODEL.md).

| Scenario | K | Formation | Protocol | Purpose |
|---|---|---|---|---|
| `dev_k2` | 2 | v_shape | ble | Minimal dev/debug case |
| `gate_k6` | 6 | encirclement | ble | Primary phase-gate scenario |
| `gate_k6_column` | 6 | column | ble | Formation confusion-matrix variant |
| `gate_k6_diamond` | 6 | diamond | ble | Formation confusion-matrix variant; **demo2 Run A** |
| `gate_k6_dispersed` | 6 | dispersed | ble | Formation confusion-matrix variant |
| `gate_k6_vshape` | 6 | v_shape | ble | Confusion-matrix variant; v_shape is the known-hard class |
| `gate_k6_wifi` | 6 | column | wifi | Multi-protocol; **demo2 Run B** |
| `gate_k6_drone` | 6 | v_shape | drone_ocusync | Deliberately hard formation on the hardest protocol; **demo2 Run C** |
| `sweep_k4` | 4 | encirclement | ble | Leading-4 subset sweep |
| `sweep_k8` | 8 | ring | ble | Procedural layout, not an STGT-recognized class |

## The `demo2` review pipeline

`demo2/` is instrumentation for reviewing a run after the fact: it captures
**every field from the actual live topics** (no synthetic/parallel data path)
into 12 numbered JSON files per run, one per pipeline stage, under
`demo2/runs/<run_id>/`:

```
00_run_manifest   01_input_rf      02_router        03_embeddings
04_clustering     05_observables   06_localization  07_association
08_formation      09_prediction    10_narration     11_run_summary
```

Three runs are captured, each telling a different, deliberately chosen story:

| Run | Scenario | Story |
|---|---|---|
| `runA_ble_nominal` | `gate_k6_diamond` | Everything works: high-confidence baseline on the easiest protocol/formation pairing. |
| `runB_wifi_column` | `gate_k6_wifi` | Proves the protocol router isn't decorative — a genuinely different frozen encoder engages for real WiFi devices. |
| `runC_drone_vshape` | `gate_k6_drone` | **Honest degradation**, not a failure: v-shape is a known-hard formation class, and drone-tier embeddings are less separable than BLE's identity-trained ones. Confidence visibly drops and narration escalates rather than confidently asserting the wrong shape. This is the intended, pre-registered behavior — see `demo2/docs/run_selection_notes.md` and `NOISE_MODEL.md` §8. |

To capture a new run yourself, once the pipeline (matching scenario) is up
and settled:

```bash
python3 demo2/scripts/capture_run.py --scenario gate_k6_diamond --run-id my_run --duration 60
```

## The AR headset client (separate repo)

The Meta Quest AR visualization is a **separate repository**, not part of
this one: it only ever talks to this pipeline over rosbridge/websocket
(`/scene_update`, `/formation`, `/narration`) — it has no direct code
dependency on anything here, so it's versioned separately.

If you're a teammate looking for "where's the headset code," it isn't in
`RF_Swarm` — ask whoever has the AR-visualization repo link, or check the
GitHub org for it. `capstone_ws/client/` in *this* repo is just the small
JS bridge/parsing helpers, not the AR scene itself.

## Project history

This was built in phases, each with a hard gate before moving on. In brief:

0. Message contract frozen, stub nodes wired end-to-end.
1. Scripted swarm kinematics + real 5-receiver TDoA array geometry.
2. RF observable generator + pre-registered receiver noise model.
3. Real ESP32-C3 BLE embeddings + eigengap K-estimation (base station).
4. TDoA init + iterated EKF track fusion (with a real finding: per-receiver
   clock bias is unobservable from a compact swarm's TDoA alone — fixed via
   an informative geometric prior, not by adding more sensors).
5. STGT formation classifier integration + narration bucket routing.
6. AR bridge wired end-to-end over `wss://`.
7. Real multi-protocol RFF: WiFi/drone tiers wired through the same router,
   `demo2` review instrumentation, AR visual encoding refinements.

Each phase's detailed status/findings live in project notes; the docs listed
below are the ones that matter for *running* the system, not the full phase
history.

## Known gaps — read this before demoing

Being upfront about what's *not* fully proven, so nobody gets surprised in a
review:

- **AR headset visual changes are untested in-headset.** The latest AR client
  edits (uncertainty ring, formation confidence display) were verified by
  code review only — no browser/headset was available to load and check them
  visually. Test in-headset before relying on them in a live demo.
- **Run B and Run C's LLM narration tier did not fire.** Both captures hit
  the narration node's GPU-memory safety guard (another process was
  legitimately using shared GPU memory at capture time) and fell back to the
  template tier only. The guard worked exactly as designed — it halts rather
  than crashing or degrading silently — but it means the `10_narration.json`
  artifacts for B/C don't demonstrate the LLM tier. Re-run when the GPU is
  free if you need that evidence.
- **The Qwen2.5-7B base model (~15GB) is not vendored** into this repo (see
  `DEPENDENCY_MANIFEST.md` §5) — only the LoRA adapter is. You need the base
  model from HuggingFace separately to run the LLM narration tier at all.
- **`wifi`/`drone_ocusync` scenarios need external data/tools** (`RFFP_ROOT`,
  `DLMODEL_ROOT`, see above) that aren't part of this repo. Every `ble`
  scenario is fully self-contained.

## Doc index

| Doc | What it's for |
|---|---|
| [`capstone_msgs_spec.md`](capstone_msgs_spec.md) | Every message type, field by field |
| [`reconciliation_answers.md`](reconciliation_answers.md) | Why `/scene_update` is JSON-over-String, not a typed msg; ENU→Y-up formula |
| [`DEPENDENCY_MANIFEST.md`](DEPENDENCY_MANIFEST.md) | Provenance/checksums for every vendored or copied artifact |
| [`README_INTEGRATED_SYSTEM.md`](README_INTEGRATED_SYSTEM.md) | Earlier integration notes; directory tree + launch commands |
| [`capstone_ws/src/capstone_scenarios/NOISE_MODEL.md`](capstone_ws/src/capstone_scenarios/NOISE_MODEL.md) | Full pre-registered physics derivation for every protocol's measurement noise |
| [`capstone_ws/src/capstone_base/FINGERPRINT_METHOD.md`](capstone_ws/src/capstone_base/FINGERPRINT_METHOD.md) | Eigengap K-estimation + merge-tau method detail |
| [`vendor/src/swarm_intent/stgt/README.md`](vendor/src/swarm_intent/stgt/README.md) | What was and wasn't vendored from the STGT upstream repo, and why |
| [`demo2/docs/run_selection_notes.md`](demo2/docs/run_selection_notes.md) | Why Run A's formation was empirically re-chosen |
| [`capstone_ws/scripts/install_quest_ca.md`](capstone_ws/scripts/install_quest_ca.md) | Sideloading the local TLS CA onto a Quest headset |
