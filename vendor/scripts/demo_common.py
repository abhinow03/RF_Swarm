"""Shared infrastructure for the defense demo (demo_act1/2/3 + run_demo.py).

HotSwapClient is new glue code (no equivalent exists in src/swarm_intent/llm/client.py's
LocalHFClient, which only supports one fixed adapter per instance) -- the loading
pattern itself (load the 4-bit base once, PeftModel.from_pretrained/load_adapter/
set_adapter/disable_adapter to switch) was already validated in this repo by
scripts/bench_adapter_hotswap.py (AUDIT.md sec W throughput-optimization session).
generate()/generate_batch() are inherited UNCHANGED from LocalHFClient -- prompt
construction, chat-template application, and sampling logic are not reimplemented,
only the model-loading constructor differs.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "config"))

from swarm_intent.llm.client import LocalHFClient  # noqa: E402
from swarm_intent.llm.prompts import is_abstention, match_threat, match_intent  # noqa: E402
from swarm_intent import pipeline_v2  # noqa: E402

import demo_config  # noqa: E402


class HotSwapClient(LocalHFClient):
    """LocalHFClient's generate()/generate_batch() reused unmodified (inherited).
    Only __init__ differs: loads the 4-bit base model ONCE with no fixed adapter,
    then add_adapter()/use_adapter() hot-swap PEFT adapters in place on the same
    GPU-resident base weights -- same pattern scripts/bench_adapter_hotswap.py
    already measured and validated in this repo."""

    def __init__(self, model_path: str, temperature: float = 0.0,
                 max_new_tokens: int = 512, system_prompt=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=quant, device_map="auto", torch_dtype=torch.float16)
        self.model.eval()
        self._is_peft = False
        self._adapter_names = set()

    def add_adapter(self, name: str, path: str) -> None:
        """Idempotent: adding the same name twice is a no-op."""
        if name in self._adapter_names:
            return
        from peft import PeftModel
        if not self._is_peft:
            self.model = PeftModel.from_pretrained(self.model, path, adapter_name=name)
            self._is_peft = True
        else:
            self.model.load_adapter(path, adapter_name=name)
        self.model.eval()
        self._adapter_names.add(name)

    def use_adapter(self, name):
        """Context manager. name=None -> pure base model (disable_adapter());
        name=<str> -> set_adapter(name) for the remainder of the block (PEFT's
        set_adapter is sticky, not scoped, so this is a null contextmanager that
        just switches state -- matches bench_adapter_hotswap.py's own usage)."""
        if name is None:
            if self._is_peft:
                return self.model.disable_adapter()
            return contextlib.nullcontext()
        if name not in self._adapter_names:
            raise ValueError(f"adapter {name!r} not loaded -- call add_adapter() first")
        self.model.set_adapter(name)
        return contextlib.nullcontext()


def gpu_free_gib(required_gib: float = 15.0) -> float:
    """Reports current free VRAM in GiB. Raises RuntimeError (caller should
    print and halt, per LOCKED CONFIG: 'if insufficient, report and halt
    rather than attempting and crashing mid-demo') if below required_gib."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible -- cannot run the live demo")
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024 ** 3)
    if free_gib < required_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB free GPU memory, need >= {required_gib:.1f} GiB -- "
            f"halting before loading anything (see V5A2 training session's OOM crash for "
            f"why this project checks first rather than attempting and crashing)")
    return free_gib


def load_seed999_items() -> list[dict]:
    with open(demo_config.SEED999_EVAL_SET) as f:
        return json.load(f)["items"]


def load_v5a_seed999_results() -> dict:
    with open(demo_config.V5A_SEED999_RESULTS) as f:
        return json.load(f)


def pick_act1_case() -> dict:
    """Pull ONE answerable (has_ground_truth=True) case from the locked
    seed=999 eval set where v5-a's ALREADY-MEASURED real output
    (evaluation/v5a_seed999_results.json, sec AQ erratum part 3's real GPU
    run) is correct -- first match in file order, not hand-picked. Raises if
    none exists (would itself be a real finding, not something to paper over)."""
    items = load_seed999_items()
    results = load_v5a_seed999_results()
    parsed_by_case = results["parsed_by_case"]
    by_name = {it["name"]: it for it in items}

    for name, parsed in parsed_by_case.items():
        item = by_name.get(name)
        if item is None or not item.get("has_ground_truth"):
            continue
        if not isinstance(parsed, dict):
            continue
        intent = parsed.get("likely_intent", "")
        if is_abstention(intent):
            continue
        if match_threat(parsed.get("threat_level", ""), item["expected_threat"]):
            return item
    raise RuntimeError("no has_ground_truth=True case found where v5-a's already-measured "
                       "seed999 output is correct -- cannot build Act 1 without one")


def pick_bucket_case(bucket: str) -> dict:
    """First seed999 item (file order) that pipeline_v2.classify_ctx() itself
    (re-run live on the item's stored ctx/key_windows) buckets as `bucket`.

    Deliberately does NOT trust seed999_eval_set.json's own precomputed
    "bucket" field: that field was written by build_seed999_eval_set.py via
    coverage.classify_observation() on the original real STGT prediction
    dicts, a DIFFERENT entry point from classify_ctx() (which re-derives
    bucket-relevant signals by parsing the templated ctx TEXT string).
    Found live during this demo's own dry-run: at least one item stored as
    bucket="C" (Layer 3) is reclassified as bucket="A" (Layer 1) by
    classify_ctx() on its own stored ctx -- a real, previously-undocumented
    disagreement between the two classification entry points, not a demo
    bug. Since Act 2/3 call assess_ctx() (which internally calls
    classify_ctx()), selection must use the SAME function or the case
    picked to demonstrate one layer can silently route through another."""
    for item in load_seed999_items():
        if pipeline_v2.classify_ctx(item["ctx"], item["key_windows"])["bucket"] == bucket:
            return item
    raise RuntimeError(f"no seed999 item found where classify_ctx() itself returns bucket={bucket!r}")


def case_prompt(item: dict) -> str:
    from swarm_intent.inference import build_llm_prompt
    return build_llm_prompt(pipeline_v2._preds_from_key_windows(item["key_windows"]), item["ctx"], {})


def is_case_correct(parsed, item: dict) -> bool | None:
    """None if unparseable/abstained (neither correct nor incorrect -- a
    non-answer), True/False otherwise. Never silently treated as correct."""
    if not isinstance(parsed, dict):
        return None
    intent = parsed.get("likely_intent", "")
    if is_abstention(intent):
        return None
    return bool(match_threat(parsed.get("threat_level", ""), item["expected_threat"]))


# --- grounded facts (already measured elsewhere in this project -- cited, not invented) ---

FACT_BASE_IS_WEAKEST = (
    "no fine-tuning; weakest of the 5 Phase-4-compared systems on every headline metric "
    "(evaluation/phase4_baselines_scored.json via llm_finetuning/build_phase4_comparison_table.py)"
)
FACT_V2_ZERO_ABSTENTION = (
    "prior fine-tuned baseline; 0.0% correct-abstention on genuinely unanswerable cases -- "
    "always answers confidently, never hedges (AUDIT.md sec AI)"
)
FACT_V5A_HEADLINE = (
    "current production checkpoint (v5a2); 78.9% accuracy_when_answerable, "
    "memorization-consistent-with-generalization (0.0%/n=642 vs a 15% memorization-signal bar, "
    "0.74% fresh null baseline) -- passed 8/10 preregistered bars, shipped despite failing both "
    "abstention-mechanism bars (3.2% multi-hop / 1.6% oscillation correct-abstention -- see "
    "docs/CONFIDENCE_FLAG.md's caution-flag mitigation) (docs/V5_LOG.md, AUDIT.md sec AR)"
)
FACT_V5A_ZERO_ABSTENTION = (
    "0.0% correct-abstention on the 502 genuinely unanswerable (multi-hop) cases in the Phase 4 "
    "population -- v5-a never hedges either, shared with v2, not a v5-a-specific defect (AUDIT.md sec AI)"
)


def run_pipeline_case(client, class_freq: dict, ctx: str, key_windows: list) -> dict:
    """Shared by demo_act2_pipeline.py, demo_act3_live.py, and
    scripts/build_demo_webdata.py so all three presentations (terminal Act 2,
    terminal Act 3, web interface) route through pipeline_v2.assess_ctx()
    the exact same way -- classify bucket, pick the right adapter state
    (base for Layer 1's narrator, ACTIVE_ADAPTER for Layer 3, irrelevant for
    Layer 2's no-model-call guard), dispatch, and catch failures explicitly
    (never silently relabeled as success). Returns a dict, not printed."""
    from swarm_intent.coverage import BUCKET_A, BUCKET_C

    bucket_info = pipeline_v2.classify_ctx(ctx, key_windows)
    bucket = bucket_info["bucket"]
    adapter_name = None if bucket in (BUCKET_A,) else ("active" if bucket == BUCKET_C else None)

    try:
        with client.use_adapter(adapter_name):
            assessment, layer, detail = pipeline_v2.assess_ctx(
                rules_narrator_client=client, finetuned_client=client,
                ctx=ctx, key_windows=key_windows, class_freq=class_freq)
        return {"bucket": bucket, "layer": layer, "detail": detail,
                "assessment": assessment, "error": None}
    except Exception as e:
        return {"bucket": bucket, "layer": None, "detail": None,
                "assessment": None, "error": f"{type(e).__name__}: {e}"}


def banner(text: str, char: str = "=", width: int = 90) -> None:
    print(char * width)
    print(text)
    print(char * width)


def layer_banner(layer: str) -> None:
    labels = {
        pipeline_v2.LAYER_1_DETERMINISTIC: "LAYER 1 -- deterministic rule-table lookup (no LLM decision, narration only)",
        pipeline_v2.LAYER_2_GUARD: "LAYER 2 -- guard abstention (no model call at all)",
        pipeline_v2.LAYER_3_LLM: "LAYER 3 -- LLM judgment (no RULES entry can exist for this case)",
    }
    banner(labels.get(layer, layer), char="-")
