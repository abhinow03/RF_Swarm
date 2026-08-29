"""Defense demo, Act 1: three-system contrast on one real scenario.

Loads the 4-bit Qwen2.5-7B-Instruct base ONCE, then hot-swaps between no
adapter (base) / qwen-swarm-v2 / ACTIVE_ADAPTER (config/demo_config.py --
currently v5-a) via demo_common.HotSwapClient, instead of three separate
7B loads (the phase4-baseline OOM incident this project already hit once).

The scenario is pulled programmatically from the locked seed=999 eval set
(config/demo_config.SEED999_EVAL_SET): the first has_ground_truth=True case
where v5-a's ALREADY-MEASURED real output (evaluation/v5a_seed999_results.json)
is correct -- not hand-picked. Reuses build_llm_prompt/OUTPUT_SCHEMA and the
same match_threat/is_abstention scoring convention every eval script in this
project uses -- no new scoring logic here.

Usage:
    python scripts/demo_act1_contrast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "config"))

import demo_common as dc  # noqa: E402
import demo_config  # noqa: E402
from swarm_intent.llm.client import _parse_llm_response  # noqa: E402


SYSTEMS = [
    ("base", "base (no adapter)", None, dc.FACT_BASE_IS_WEAKEST),
    ("v2", "v2 (prior fine-tuned baseline)", "v2", dc.FACT_V2_ZERO_ABSTENTION),
    ("v5a", "v5a2 (current production, config/demo_config.ACTIVE_ADAPTER)", "active", dc.FACT_V5A_HEADLINE),
]


def build(client: "dc.HotSwapClient") -> dict:
    """Runs all 3 systems on the Act 1 scenario and returns a structured dict
    (used by both the terminal printer below and scripts/build_demo_webdata.py
    for the web interface's cached Act 1 payload) -- one code path, two
    presentations, no duplicated generation/scoring logic."""
    case = dc.pick_act1_case()
    prompt = dc.case_prompt(case)

    results = []
    for key, label, adapter_name, fact in SYSTEMS:
        with client.use_adapter(adapter_name):
            try:
                raw = client.generate(prompt)
                error = None
            except Exception as e:
                raw, error = None, f"{type(e).__name__}: {e}"
        parsed = _parse_llm_response(raw) if raw is not None else None
        correct = dc.is_case_correct(parsed, case) if error is None else None
        verdict = ("error" if error else
                   "correct" if correct else
                   "incorrect" if correct is False else "abstained/unparseable")
        results.append({
            "key": key, "label": label, "fact": fact, "error": error,
            "raw": raw, "parsed": parsed, "verdict": verdict,
        })

    return {
        "case_name": case["name"], "pair": case["pair"], "expected_threat": case["expected_threat"],
        "ctx": case["ctx"], "systems": results,
    }


def print_report(data: dict) -> None:
    case = data
    dc.banner(f"ACT 1: three-system contrast -- scenario {case['case_name']!r} "
             f"({case['pair'][0]} -> {case['pair'][1]}, expected_threat={case['expected_threat']})")
    print(case["ctx"])
    print()
    for sysresult in case["systems"]:
        dc.banner(sysresult["label"], char="-")
        if sysresult["error"]:
            print(f"GENERATION FAILED: {sysresult['error']}")
            print()
            continue
        print((sysresult["raw"] or "").strip()[:1000])
        print(f"\n>>> {sysresult['verdict']} -- {sysresult['fact']}")
        print()


def run(client: "dc.HotSwapClient" = None) -> dict:
    owns_client = client is None
    if owns_client:
        dc.gpu_free_gib(required_gib=15.0)
        client = dc.HotSwapClient(demo_config.BASE_MODEL, temperature=0.0)
        client.add_adapter("v2", demo_config.V2_ADAPTER)
        client.add_adapter("active", demo_config.ACTIVE_ADAPTER)

    data = build(client)
    print_report(data)

    if owns_client:
        del client
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    return data


if __name__ == "__main__":
    run()
