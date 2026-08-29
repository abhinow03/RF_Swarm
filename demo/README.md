# RF_Swarm — End-to-End Demo (recorded from a REAL run)

Everything here was **captured from one real pipeline run** (`gate_k6`, all nodes
live, GPU LLM narration on) — not mocked. It walks the full data path a swarm of
6 RF emitters takes through the system: **raw RF → embeddings/fingerprint →
localization (TDoA + EKF) → formation → path prediction → LLM narration → AR client.**

Start with `figures/` for the picture, then read the folders in order.

```
demo/
├── figures/                     ← LOOK HERE FIRST (PNGs)
│   ├── 04_pipeline_dataflow.png   the whole pipeline at a glance
│   ├── 01_scene_topdown.png       6-drone swarm flying through the 5-receiver array; EKF estimates vs truth
│   ├── 02_localization_error.png  EKF error over time (~15 m, the documented clock-limit)
│   └── 03_formation_timeline.png  STGT formation classification + when real LLM narrations fired
│
├── 01_inputs/                   ← WHAT THE SYSTEM IS GIVEN / SEES
│   ├── scenario.yaml              the scenario config (geometry, RF model, clock model)
│   ├── receiver_array.csv         the 5 fixed TDoA receivers (ENU + published clock uncertainty)
│   ├── ground_truth_emitters.csv  the 6 true emitter trajectories — THE ANSWER we recover (never seen by estimators)
│   ├── rf_observations_sample.csv the raw measurements the estimators actually see: ToA / RSSI / FDoA per (burst, receiver)
│   └── embeddings_sample.csv      blind 128-D BLE encoder vectors (identity signal, no unit label)
│
├── 02_multilateration/          ← THE LOCALIZATION MATH, CONCRETE
│   ├── tdoa_worked_example.json   ONE burst end-to-end: per-receiver ToA → range-differences → closed-form
│   │                              TDoA init → EKF-refined estimate → ground truth → error
│   ├── tracks_vs_truth.csv        every track over time: EKF estimate vs true emitter + error + GDOP
│   └── localization_summary.json  RMSE / mean / p90, per-track error, the blind→true association map
│
├── 03_model_io/                 ← EACH MODEL'S RECORDED INPUT → OUTPUT
│   ├── fingerprint_io.json         base station: accumulated tracks → eigenvalues, K̂, per-cluster assignments
│   ├── ekf_io.json                 EKF: range-difference measurement + clock prior → track state (+ GDOP)
│   ├── stgt_formation_io.json      STGT: 50×6×3 pose sequence → formation class + all 8 class probabilities
│   └── llm_narration_io.json       LLM: the tactical-context STRING in → the generated narration OUT (real GPU output)
│
├── 04_outputs/                  ← WHAT THE SYSTEM PRODUCES
│   ├── tracks.csv                  EKF localization output (position/velocity/gdop/status)
│   ├── formation_timeline.csv      STGT classification over the whole run
│   ├── predictions_sample.json     CV-baseline path prediction (waypoints + growing uncertainty)
│   ├── narration_log.txt           every narration, incl. the 2 real GPU-LLM ones
│   └── scene_update_sample.json    the EXACT JSON the AR/Quest client receives (tracks-only, Y-up)
│
├── _run_raw/                     raw per-topic capture (JSONL) — the full firehose, if you want everything
└── _tools/                       the record/build/plot scripts (reproduce with any run)
```

## The one-paragraph story
The sim emits 6 drones flying as a moving formation through a fixed 5-receiver
array. Each receiver reports a bias-corrupted **time-of-arrival** (`01_inputs`).
The base station clusters **blind BLE embeddings** to discover *how many* emitters
and give each a stable ID (`03_model_io/fingerprint_io.json`). The **joint EKF**
multilaterates position from time-of-arrival *differences* while simultaneously
estimating the receiver clock bias as a state (`02_multilateration`) — it recovers
the swarm to **~15 m** (clock-prior-limited; see `02_.../localization_summary.json`).
The **STGT classifier** reads the EKF track sequence and names the formation
(`03_.../stgt_formation_io.json`); a confident formation routes to the **LLM**,
which writes a tactical narration (`03_.../llm_narration_io.json`, real Qwen-7B
output). The **bridge** flattens tracks to the client wire format
(`04_.../scene_update_sample.json`).

## Key numbers from this run
- Localization RMSE **~15 m**; association **bijective** (each discovered ID ↔ one true emitter).
- Base station **K̂ = 6** (correct count); per-emitter fingerprint confidence 0.84–0.98.
- 2 real GPU LLM narrations generated (see `04_outputs/narration_log.txt`), e.g.
  *"Multiple UAVs are forming an encircling pattern around a target, indicating a high-level threat."*
- Formation classification cycles encirclement/dispersed/v_shape — an **honest** artifact
  of feeding ~15 m-noisy tracks into a classifier trained on clean geometry (documented, not a bug).

## Reproduce
```bash
source capstone_ws/activate.sh
# terminal A: pipeline (no narrate) + GPU narrate
bash capstone_ws/scripts/run_pipeline.sh gate_k6 sim rff base track formation predict bridge
bash capstone_ws/scripts/run_narrate_llm.sh
# terminal B: record, then build + plot
python3 demo/_tools/demo_recorder.py demo/_run_raw 100
python3 demo/_tools/demo_build.py && python3 demo/_tools/demo_figures.py
```
