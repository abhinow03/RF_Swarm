"""Single source of truth for the defense demo's model-swap seam.

Every demo script (scripts/demo_act1_contrast.py, demo_act2_pipeline.py,
demo_act3_live.py, run_demo.py) imports ACTIVE_ADAPTER from here instead of
hardcoding an adapter path. Swapping the demo's "current production LLM"
from v5-a to v5a2 (once v5a2 passes its preregistered bars, see
docs/PREREGISTRATION_V5A2.md) is a one-line edit to ACTIVE_ADAPTER below --
nothing else in the demo code may name a "v5_sft*" path directly.

ACTIVE_ADAPTER_TRAIN_FILE travels WITH ACTIVE_ADAPTER: pipeline_v2.py's
Layer-3 prior correction (_rescore_and_correct -> scoped_correct) uses the
active adapter's OWN training-file class frequency (the same per-adapter-
frequency convention pipeline_v2.default_class_freq() uses for v3b-fix) --
if you change ACTIVE_ADAPTER you must change ACTIVE_ADAPTER_TRAIN_FILE too,
or the prior correction will be scored against the wrong distribution.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # resolved from HF cache (revision a09a3545); see DEPENDENCY_MANIFEST.md

# VENDORED into RF_Swarm (Part-2 packaging): ACTIVE_ADAPTER repointed (absolute) at
# the self-contained in-RF copy under models/. ACTIVE_ADAPTER_TRAIN_FILE is NOT used
# by the live narration path (run_pipeline_case -> assess_ctx), so it is left as a
# REPO-relative path and simply never loaded.
_RF_ROOT = "/home/pw26_akp_01/RF_Swarm"
# --- THE SWAP SEAM (repointed to in-RF copy) ---
ACTIVE_ADAPTER = _RF_ROOT + "/models/llm_adapter/v5_sft_v5a2"
ACTIVE_ADAPTER_TRAIN_FILE = str(REPO / "data" / "sft_train_v5a2_train.jsonl")  # not runtime-loaded
# --------------------------------------------------------------------------------

# Act 1's fixed historical-baseline contrast point (the prior fine-tuned
# system, AUDIT.md sec AI: 0.0% correct-abstention, same as v5-a) -- NOT part
# of the swap seam, always v2 regardless of which adapter ACTIVE_ADAPTER points at.
V2_ADAPTER = str(REPO / "adapters" / "qwen-swarm-v2")

SEED999_EVAL_SET = str(REPO / "evaluation" / "seed999_eval_set.json")
V5A_SEED999_RESULTS = str(REPO / "evaluation" / "v5a_seed999_results.json")

PROTECTED_CHECKPOINT_DIR = str(REPO / "checkpoints" / "v5_sft_v5a_PROTECTED")
