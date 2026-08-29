# RF_Swarm — DEPENDENCY MANIFEST (self-contained packaging)

Records every external artifact the integrated pipeline depends on at runtime,
copied into `RF_Swarm/` so the system is self-contained. **Originals are NOT
modified or moved.** Copied 2026-08-26.

Peer code is copied from the **pinned commit**, not the peer working tree. Pinned
commit and the peer's current HEAD were checked: **identical** (0 commits ahead),
so no newer peer work is being deliberately excluded.

Layout:
- `vendor/`  — copied peer CODE (pinned), mirrors the peer `src/scripts/config` layout
- `models/` — copied model WEIGHTS / adapters
- `data/ble_embeddings/` — BLE embedding pool (already in-repo since Phase 3)

---

## 1. Vendored peer code — `vendor/`
- **Source repo:** `/home/pw26_akp_01/aadhya/uav-llm`
- **Commit (pinned == HEAD):** `69ac2b69015b7d8e37b388acb3443f5073c0eae3`
- **Working tree at copy time:** clean for all copied paths (running code == pinned commit)
- **Method:** `git archive 69ac2b6 <path>` / `git show 69ac2b6:<path>` (from the commit, not the worktree)

| In-RF path | Source path @ pin | Notes |
|---|---|---|
| `vendor/src/swarm_intent/` (17 .py) | `src/swarm_intent/` | formations, stgt/{config,model,inference}, coverage, context_spec, pipeline_v2, llm/*, … |
| `vendor/scripts/` (54 .py) | `scripts/` | runtime-critical: **demo_common.py** (HotSwapClient, run_pipeline_case). Rest copied for transitive-import safety. |
| `vendor/llm_finetuning/build_sft_dataset.py` | `llm_finetuning/build_sft_dataset.py` | imported by pipeline_v2 (RULES, synth_context) via its own `sys.path.insert(REPO_ROOT/llm_finetuning)` |
| `vendor/config/demo_config.py` | `config/demo_config.py` | **EDITED**: `ACTIVE_ADAPTER` repointed to in-RF `models/llm_adapter/v5_sft_v5a2` (absolute); `BASE_MODEL` unchanged (HF id). |
| `vendor/data/sft_train_final_abstain_fix.jsonl` | `data/sft_train_final_abstain_fix.jsonl` | read by `pipeline_v2.default_class_freq()` at model load. sha256 `4b9df3a35b019258…` |

Runtime import roots (set by `activate.sh` / `run_narrate_llm.sh`):
`vendor/src` : `vendor/scripts` : `vendor/config`. No `~/aadhya/uav-llm` on PYTHONPATH at runtime.

## 2. STGT formation classifier weights — `models/stgt/`
- **Source:** `/home/pw26_akp_01/aadhya/uav-llm/swarm_data/` (gitignored working-tree artifacts — recorded by checksum + mtime, not a git object)
- **Consumed by:** `capstone_formation` (default `model_dir` now `RF_Swarm/models/stgt`)

| file | sha256 | src mtime |
|---|---|---|
| `best_model.pt` | `18fc201d5a419ff2fb0cfb66a60810af77a9ed52969f2996f57c952d1306a01b` | 2026-08-24 |
| `train_mean.npy` | `3d77ee166bb7f15466f579d2e6bacdbb5239c01cdb91f2fcee60d3981bd342fa` | 2026-08-03 |
| `train_std.npy` | `ac13db055a3e0aadc27fd9cff25cb77464b191bb7ac1af326c6cf96ef2229863` | 2026-08-03 |

`best_model.pt` embeds its own matched `reg_mean/reg_std` (Phase-5 validated; val_acc 0.868). No separate reg-stat files needed.

## 3. LLM narration LoRA adapter — `models/llm_adapter/v5_sft_v5a2/`
- **Source:** `/home/pw26_akp_01/aadhya/uav-llm/checkpoints/v5_sft_v5a2/` (gitignored; mtime 2026-08-18)
- **Consumed by:** `capstone_narrate` bucket-A path (`demo_config.ACTIVE_ADAPTER`), loaded via PeftModel over the 4-bit Qwen base

| file | sha256 |
|---|---|
| `adapter_model.safetensors` | `2551e3a486dee2515fd4ca34c617c0839b131f0c35b656a34c1cfb4d1255b9f8` |
| `adapter_config.json` | `00599324214e78f64d5e9235b87dbfb55ddadd40bf30a9c4b02793f9954cc6f9` |
| `chat_template.jinja` | `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f` |

## 4. BLE embedding pool — `data/ble_embeddings/ble_xiao_outdoor_pool.npz` (ALREADY in-repo)
- In-RF since Phase 3 (no copy needed). Consumed by `capstone_rff` → `/embeddings`.
- File sha256: `8f43a792ed79bbfac80a9ad2604b962b421f707d17b514617a9f6882186f4b8b`
- **Encoder provenance (inside the npz):** `encoder_sha256 = f1d7745eaa726e6a252961015a1647bf407f4e7815c022f128a12d9f752a2ebb` — the frozen ESP32-C3 BLE encoder (canonical). The encoder checkpoint itself is NOT a runtime dependency (the pipeline replays the pool; the encoder was one-time-use for pool generation).

## 5. LLM base model — Qwen2.5-7B-Instruct  ⚠️ NOT COPIED (flagged decision)
- **Source:** HF cache `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct`
- **HF revision (pinned):** `a09a35458c702b33eeacc393d103063234e8bc28`  •  **size 15 GB**
- **Consumed by:** `capstone_narrate` bucket-A path (`demo_config.BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"`), loaded 4-bit (bitsandbytes nf4) on GPU.
- **Status:** left in the HF cache — the ONLY runtime artifact not physically inside `RF_Swarm/`. It is a stock, revision-pinned HF model (reproducible), and copying 15 GB onto a 90%-full shared disk is a resource-heavy call, so it is flagged for explicit go-ahead. To make RF fully self-contained:
  ```bash
  mkdir -p RF_Swarm/models/qwen2.5-7b-instruct
  cp -rL ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/* \
         RF_Swarm/models/qwen2.5-7b-instruct/
  # then set vendor/config/demo_config.py: BASE_MODEL = "/home/pw26_akp_01/RF_Swarm/models/qwen2.5-7b-instruct"
  ```

---

## Functional-equivalence verification (copies == originals)
- **STGT:** `gate_confusion.py` re-run on `models/stgt` + vendored `swarm_intent` → truth-fed **5/5**, confidences identical (0.977/0.837/0.988/0.991/0.995).
- **LLM (isolation):** vendored code + copied adapter → identical generated string ("The UAVs have transitioned from a V-shape formation to an encirclement…").
- **LLM (live, fully self-contained):** pipeline under vendored `activate.sh` + `run_narrate_llm.sh` → `/narration source=llm` on GPU ("A small group of UAVs is maintaining a steady, diamond formation…"). No `~/aadhya/uav-llm` imported at runtime.
