# RF_Swarm — Final Integrated System (map)

Swarm RF-emitter localization + threat-assessment pipeline. As of 2026-08-26 this
folder is **self-contained**: every runtime code/weight dependency is inside
`RF_Swarm/` (one flagged exception: the 15 GB Qwen base model, still in the HF
cache — see `DEPENDENCY_MANIFEST.md §5`).

## What constitutes the final system

```
RF_Swarm/
├── capstone_ws/                      # THE ROS 2 workspace (all our nodes)
│   ├── src/
│   │   ├── capstone_msgs/            # frozen interface spec (17 msgs) — see capstone_msgs_spec.md
│   │   ├── capstone_scenarios/       # scenario YAMLs (gate_k6 + K=6 formation variants) + NOISE_MODEL.md
│   │   ├── capstone_sim/             # sim: truth kinematics + RF observable/clock model  -> /rf_observations /receivers
│   │   ├── capstone_rff/             # replays BLE embeddings (blind)                       -> /embeddings
│   │   ├── capstone_base/            # eigengap+merge fingerprinting + assoc publishing     -> /fingerprints /associations
│   │   ├── capstone_track/           # closed-form TDoA init + joint iterated EKF           -> /tracks
│   │   ├── capstone_formation/       # STGT formation classifier (torch/CPU)               -> /formation
│   │   ├── capstone_predict/         # CV baseline path predictor                          -> /predictions
│   │   ├── capstone_narrate/         # ctx router + LLM narration (bucket-A GPU)           -> /narration
│   │   ├── capstone_bridge/          # Option-C /scene_update (tracks-only, ENU->Y-up)     -> /scene_update
│   │   └── capstone_common/          # HealthReporter + QoS helpers
│   ├── activate.sh                   # env: jazzy + .venv + VENDORED peer code + overlay + ROS_DOMAIN_ID=42
│   ├── client/                       # rf_bridge.js (roslibjs wss drop-in) + test.html  [for Quest, tomorrow]
│   ├── certs/                        # mkcert CA + leaf cert for rosbridge wss           [for Quest, tomorrow]
│   ├── tools/mkcert                  # mkcert binary
│   └── scripts/
│       ├── run_pipeline.sh           # launch the pipeline (clean start/teardown)
│       ├── run_narrate_llm.sh        # run narrate under the LLM env (GPU Qwen)
│       └── run_rosbridge_wss.sh      # rosbridge over wss                                [for Quest, tomorrow]
├── vendor/                           # VENDORED peer code, pinned 69ac2b6 (src/scripts/config/llm_finetuning/data)
├── models/
│   ├── stgt/                         # STGT classifier weights (best_model.pt + norm stats)
│   └── llm_adapter/v5_sft_v5a2/      # LLM LoRA adapter
├── data/ble_embeddings/              # BLE embedding pool (in-repo since Phase 3)
├── DEPENDENCY_MANIFEST.md            # every copied artifact: source path, commit, checksum
└── README_INTEGRATED_SYSTEM.md       # this file
```

## Data path
`capstone_sim` → `/rf_observations` → `capstone_rff` `/embeddings` → `capstone_base`
`/fingerprints`+`/associations` → `capstone_track` `/tracks` → { `capstone_formation`
`/formation`, `capstone_predict` `/predictions`, `capstone_narrate` `/narration` } →
`capstone_bridge` `/scene_update` (+ `/formation` `/narration` as separate latched
topics) → rosbridge wss → AR/Quest client.

## Launch (full pipeline, gate_k6)
`capstone_rff` (real BLE IQ → RFFP router + BLE encoder) and `capstone_narrate` (GPU
LLM) both need the CUDA env, so they run via their own launchers; the rest runs
under the normal env via run_pipeline.sh.
```bash
source capstone_ws/activate.sh
bash capstone_ws/scripts/run_pipeline.sh gate_k6 sim base track formation predict bridge  # normal env
bash capstone_ws/scripts/run_rff_real.sh gate_k6      # rff: real IQ -> RFFP router+encoder (GPU pre-encode, then frees GPU)
bash capstone_ws/scripts/run_narrate_llm.sh           # narrate: bucket-A Qwen-7B on GPU
```
`capstone_rff` streams real BLE IQ (`/home/docker/…/ext_data/ble_xiao`) through the
RFFP protocol router + frozen BLE encoder (`~/Desktop/RFFP_DEPLOY`) — replacing the
old embedding-pool replay (see `DEPENDENCY_MANIFEST` / rffp-integration notes).

## Gates / verification (all self-contained)
- `capstone_track/scripts/validate_ekf.py` — EKF localization RMSE/NEES (offline, multi-seed)
- `capstone_formation/scripts/gate_confusion.py` — STGT truth-fed confusion (loads `models/stgt`)
- `capstone_predict/scripts/cv_baseline_eval.py` — CV path RMSE
- `client/parsers_test.js` / `headless_test.mjs` — client bridge

## Two environments (deliberate)
- **project `.venv`** (`activate.sh`): CPU torch + torch_geometric for the STGT classifier; everything except narrate-GPU.
- **`~/aadhya/uav-llm/.venv`** (`run_narrate_llm.sh`): CUDA torch + transformers + bitsandbytes for the 4-bit Qwen LLM. Only `capstone_narrate` uses it; it reaches rclpy/capstone_msgs + VENDORED peer code via the sourced overlays. (This one env is not duplicated into RF — it's a heavyweight CUDA stack; the CODE it runs IS vendored.)
```
