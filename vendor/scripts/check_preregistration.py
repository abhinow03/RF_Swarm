"""
Checks an eval_sft_v5.py results JSON against the success bars committed in
docs/PREREGISTRATION.md, before any real v5-a number existed to shape them.

Bars checked (map directly onto eval_sft_v5.py's OUTPUT_SCHEMA_DOC fields):
  - threat accuracy, pooled, when-answerable        >= 55.0%
  - threat accuracy, `low` (per-class)               >= 65.0%
  - over-abstention rate                             <= 15.0%
  - schema-validity rate                             >= 95.0%
  - escalation direction: under-escalation must remain the LARGER of the two
    directions (qualitative, not a fixed numeric threshold -- reported either
    way, not silently passed/failed)
  - pair accuracy, pooled, when-answerable          >= 55.0% -- HARDENING AUDIT
    step 2 FIX: this was previously NOT COMPUTABLE (eval_sft_v5.py's schema had
    no field measuring whether the model identified the (form_a,form_b) pair
    independent of threat_level match). Now computed via eval_sft_v5.evaluate()'s
    `pair_accuracy` field: a PROXY (likely_intent match against
    RULES[true_pair]'s intent -- more exacting than threat-tier match, since 49
    RULES pairs collapse onto only 4 threat tiers but many more distinct
    intents), NOT literal pair-string matching (no such field exists in
    OUTPUT_SCHEMA; getting one would require adding it and re-generating every
    system's output -- not attempted). Reported with the proxy caveat attached,
    every time, not silently presented as if it were the literal metric.

NOT checked here -- flagged explicitly, not silently skipped or faked:
  - the memorization-vs-generalization overlap-rate check (docs/
    PREREGISTRATION.md's other section): needs the raw training corpus +
    per-case generated text, not just this aggregate results JSON -- see
    llm_finetuning/score_memorization.py for that check.

Usage:
    python scripts/check_preregistration.py logs/eval_v5_results.json
"""
from __future__ import annotations

import argparse
import json
import sys

BARS = {
    "threat_accuracy_pooled_when_answerable": {"op": ">=", "value": 0.55},
    "threat_accuracy_low_per_class": {"op": ">=", "value": 0.65},
    "over_abstention_rate": {"op": "<=", "value": 0.15},
    "schema_validity_rate": {"op": ">=", "value": 0.95},
    "pair_accuracy_pooled_when_answerable": {"op": ">=", "value": 0.55},
}


def extract_metrics(results: dict) -> dict:
    """Pulls the checkable numbers out of eval_sft_v5.py's results schema.
    Missing/None fields are surfaced as None, not silently defaulted to 0 --
    a genuinely missing metric must not be scored as a pass or a fail."""
    answerability = results.get("answerability", {})
    per_class = results.get("per_class_threat_accuracy", {})
    escalation = results.get("escalation", {})
    schema = results.get("schema_validity_rate", {})
    pair = results.get("pair_accuracy", {})

    low = per_class.get("low", {})
    return {
        "threat_accuracy_pooled_when_answerable": answerability.get("accuracy_when_answerable"),
        "threat_accuracy_low_per_class": low.get("accuracy"),
        "over_abstention_rate": answerability.get("over_abstention_rate"),
        "schema_validity_rate": schema.get("rate"),
        "pair_accuracy_pooled_when_answerable": pair.get("accuracy"),
        "under_escalated": escalation.get("under_escalated"),
        "over_escalated": escalation.get("over_escalated"),
    }


def check_bar(name: str, value, bar: dict) -> str:
    if value is None:
        return "MISSING"
    if bar["op"] == ">=":
        return "PASS" if value >= bar["value"] else "FAIL"
    if bar["op"] == "<=":
        return "PASS" if value <= bar["value"] else "FAIL"
    raise ValueError(f"unknown op {bar['op']}")


def check_escalation_direction(under, over) -> str:
    """Qualitative bar: under-escalation must remain the LARGER direction
    (this project's dominant historical failure mode). Reported as its own
    row regardless of outcome -- an over-escalation-dominant result is not
    silently absorbed, it's a real finding."""
    if under is None or over is None:
        return "MISSING"
    if under == 0 and over == 0:
        return "N/A (no escalation errors at all)"
    return "PASS (under >= over, matches historical pattern)" if under >= over else \
          "FAIL (over-escalation now dominates -- a real, reportable shift from every prior system)"


def run_check(results: dict) -> tuple[list[dict], str]:
    metrics = extract_metrics(results)
    rows = []
    for name, bar in BARS.items():
        value = metrics[name]
        status = check_bar(name, value, bar)
        if name == "pair_accuracy_pooled_when_answerable" and status != "MISSING":
            status += " (PROXY: likely_intent match, not literal pair-string matching -- see module docstring)"
        rows.append({"bar": name, "target": f"{bar['op']} {bar['value']:.1%}",
                    "actual": f"{value:.1%}" if value is not None else "n/a", "status": status})

    esc_status = check_escalation_direction(metrics["under_escalated"], metrics["over_escalated"])
    rows.append({"bar": "escalation_direction_under_ge_over",
                "target": "under_escalated >= over_escalated",
                "actual": (f"under={metrics['under_escalated']:.1%}, over={metrics['over_escalated']:.1%}"
                          if metrics["under_escalated"] is not None else "n/a"),
                "status": esc_status})

    # MISSING blocks overall PASS just like FAIL does -- an incomplete results file must
    # never be reported as having cleared the bars.
    overall = "PASS" if all(r["status"] == "PASS" or r["status"].startswith("PASS")
                           for r in rows) else "FAIL"
    return rows, overall


def print_report(rows: list[dict], overall: str):
    print("| bar | target | actual | status |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['bar']} | {r['target']} | {r['actual']} | {r['status']} |")
    print(f"\nOVERALL: {overall}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", help="path to an eval_sft_v5.py results JSON")
    args = ap.parse_args()

    with open(args.results_json) as f:
        results = json.load(f)

    rows, overall = run_check(results)
    print_report(rows, overall)
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
