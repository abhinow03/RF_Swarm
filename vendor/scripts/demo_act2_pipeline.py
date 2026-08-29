"""Defense demo, Act 2: the architecture story -- pipeline_v2's 3-layer routing.

Reuses src/swarm_intent/pipeline_v2.py's actual assess_ctx() / classify_ctx()
unmodified -- no routing or scoring logic is reimplemented here, only the
model-loading (one hot-swapped client, see demo_common.HotSwapClient) and
presentation (layer-attribution banners) are new.

DELIBERATE DEVIATION from pipeline_v2's usual eval wiring (disclosed, not
silent): eval_pipeline_v2.py normally routes Layer 3 to the v3b-fix adapter.
This demo instead routes Layer 3 to config/demo_config.ACTIVE_ADAPTER (v5-a
now, v5a2 later) -- the whole point of the demo's swap seam is to show the
CURRENT production LLM inside the real pipeline architecture, not v3b-fix.
class_freq (pipeline_v2's per-adapter prior-correction frequency table) is
read from ACTIVE_ADAPTER_TRAIN_FILE to match, per the same per-adapter-
frequency convention pipeline_v2.default_class_freq() uses for v3b-fix.

Cases are pulled programmatically from the locked seed=999 eval set, one per
bucket letter (A=Layer 1 dict hit, B=Layer 2 guard, C=Layer 3 LLM) -- first
match in file order, not hand-picked.

Usage:
    python scripts/demo_act2_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "config"))

from swarm_intent import pipeline_v2  # noqa: E402
from swarm_intent.coverage import BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402

import demo_common as dc  # noqa: E402
import demo_config  # noqa: E402

BUCKET_LABELS = {BUCKET_A: "Layer 1 dict hit", BUCKET_B: "Layer 2 guard", BUCKET_C: "Layer 3 LLM"}


def build_one_case(client, class_freq: dict, item: dict) -> dict:
    """Pure data, no printing -- reused by print_report() below and
    scripts/build_demo_webdata.py for the web interface's cached Act 2 payload."""
    result = dc.run_pipeline_case(client, class_freq, item["ctx"], item["key_windows"])
    return {
        "case_name": item["name"], "pair": item.get("pair"),
        "has_ground_truth": item["has_ground_truth"], "ctx": item["ctx"],
        "bucket": result["bucket"], "bucket_label": BUCKET_LABELS.get(result["bucket"], result["bucket"]),
        "layer": result["layer"], "detail": result["detail"],
        "assessment": result["assessment"], "error": result["error"],
    }


def build(client: "dc.HotSwapClient") -> list[dict]:
    class_freq = pipeline_v2._train_class_freq(demo_config.ACTIVE_ADAPTER_TRAIN_FILE)
    cases = [dc.pick_bucket_case(b) for b in (BUCKET_A, BUCKET_B, BUCKET_C)]
    return [build_one_case(client, class_freq, item) for item in cases]


def print_one_case(c: dict) -> None:
    dc.banner(f"CASE {c['case_name']!r} ({c['bucket_label']}) "
             f"pair={c['pair']} has_ground_truth={c['has_ground_truth']}")
    print(c["ctx"])
    print()

    if c["error"]:
        print(f"PIPELINE FAILED: {c['error']}")
        print()
        return

    layer, detail, assessment = c["layer"], c["detail"], c["assessment"]
    dc.layer_banner(layer)
    if layer == pipeline_v2.LAYER_1_DETERMINISTIC:
        print(f"rule-table key: {detail['rules_key']} -- RULES[{tuple(detail['rules_key'])}] "
             f"determined threat/intent/action directly, no LLM decision involved")
        if detail["llm_deviation"]:
            print(f"(narrator LLM tried to deviate on {list(detail['llm_deviation'])} -- overwritten, logged)")
    elif layer == pipeline_v2.LAYER_2_GUARD:
        print("guard reason(s): " + "; ".join(pipeline_v2.GUARD_REASON_TEXT.get(r, r)
                                              for r in detail["guard_reasons"]))
    else:
        print(f"subtype: {detail['subtype']} -- no RULES entry can exist for this case, "
             f"ACTIVE_ADAPTER's judgment is load-bearing here")
        corr = detail["correction"]
        if corr.get("applied"):
            print(f"prior-corrected threat_level: {corr['corrected_argmax']}")

    print()
    print("final assessment:")
    for k in ("threat_level", "likely_intent", "recommended_action", "confidence_in_assessment"):
        print(f"  {k}: {assessment.get(k)}")
    print()


def run(client: "dc.HotSwapClient" = None) -> list[dict]:
    dc.banner("ACT 2: pipeline_v2's 3-layer architecture -- one case per layer")

    owns_client = client is None
    if owns_client:
        dc.gpu_free_gib(required_gib=15.0)
        client = dc.HotSwapClient(demo_config.BASE_MODEL, temperature=0.0)
        client.add_adapter("active", demo_config.ACTIVE_ADAPTER)

    data = build(client)
    for c in data:
        print_one_case(c)

    if owns_client:
        del client
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    return data


if __name__ == "__main__":
    run()
