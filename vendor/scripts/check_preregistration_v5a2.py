"""
Checks a v5a2 eval results JSON (same schema as evaluation/phase4_v5a_results.json --
answerability/escalation/per_class_threat_accuracy/schema_validity_rate/parsed_by_case) against
the bars locked in docs/PREREGISTRATION_V5A2.md, BEFORE any v5a2 result exists to shape them.

Every bar traces to a live, non-zero assertion here -- none is a table entry only. Missing
fields/files report MISSING, never a silent PASS.

Bars (see docs/PREREGISTRATION_V5A2.md section 2 for full reasoning):
  a. threat_accuracy_pooled_when_answerable      >= 55.0%
  b. pair_accuracy_pooled_when_answerable (REAL) >= 45.0%  -- self-computed via
     literal_pair_extraction.py, NOT results['pair_accuracy'] (that field is the old proxy).
  c. ceiling_normalized_accuracy                 >= 85.0%  (denominator: locked same-population
     ceiling 0.830, scripts/rule0_am_reproduce_headline.py -- see erratum part 3)
  d. over_abstention_rate                        <= 15.0%
  f. correct_abstention_rate_multi_hop           >= 25.0%  -- SEPARATE from oscillation, never pooled.
  f. correct_abstention_rate_oscillation         >= 20.0%
  g. escalation_direction (under_escalation_rate) <= 25.0%  -- real numeric ceiling.
  i. regression_vs_v5a: threat_accuracy          >= 73.9% (v5-a's 78.9% - 5.0pp)
     regression_vs_v5a: pair_accuracy (REAL)     >= 55.6% (v5-a's REAL 63.6% - 8.0pp -- WIDER
     tolerance than threat's, see erratum part 3: measuring the SAME frozen v5-a model on two
     different-but-similar populations produced a 5.7pp pair_accuracy gap on its own, which
     would itself have tripped a 5pp tolerance with zero real model change -- the tolerance is
     widened to stay safely above this measured population-selection noise floor.)
  j. memorization_vs_generalization              overlap rate < 15.0% signal bar, compared
     against the FRESH null-hypothesis baseline 0.74% (83/11,191) -- v5a2's OWN corpus-derived
     chance baseline, NOT v5-a's historical measured overlap (0.5%, n=644).

NOT a PASS/FAIL bar (dropped, reported as a diagnostic only):
  h. schema_validity_rate -- shown to pass at ~100% even for the untrained base model
     (scripts/rule0_audit_2026_08_13.py), does not discriminate capability.
  e. correct_abstention_rate_pooled (formerly "under_abstention_rate", RENAMED and DROPPED
     as a scored bar per erratum part 1) -- proven mathematically identical to the n-weighted
     average of bar f's two components. Still computed and reported, never scored.

POPULATION (erratum part 3, docs/PREREGISTRATION_V5A2.md): ALL bars now score against
evaluation/seed999_eval_set.json (seed=999, built by llm_finetuning/build_seed999_eval_set.py
from eval_data/LOCKED_seed999_FINAL.json's already-existing positions). This supersedes
erratum part 2's two-population split (a/b/c/g/i on phase4_eval_set.json seed=4321, f on
seed=999) -- v5-a's own accuracy on the two populations was re-measured and found NOT
statistically significantly different (two-proportion z-test: threat p=0.22, pair p=0.065,
n~500 each; Wilson 95% CIs overlap on both), so unification was judged safe, with the pair_
accuracy regression tolerance widened as the direct, disclosed consequence of that measurement
(see above). evaluation/phase4_eval_set.json is no longer read by this script.

Usage:
    python scripts/check_preregistration_v5a2.py <v5a2_seed999_results.json> [--memorization <overlap_results.json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.ground_truth_abstention import classify_trajectory_ground_truth, MULTI_HOP, OSCILLATION  # noqa: E402
from swarm_intent.llm.prompts import is_abstention  # noqa: E402
from literal_pair_extraction import extract_literal_pair, true_pair_from_chain  # noqa: E402

# ALL bars score against THIS file (seed=999) -- erratum part 3, full population unification.
SEED999_EVAL_SET = REPO / "evaluation" / "seed999_eval_set.json"

SAME_POPULATION_CEILING = 0.830  # scripts/rule0_am_reproduce_headline.py, live-reproduced on
                                 # LOCKED_seed999_FINAL.json, n=494 pair-eligible -- erratum part 3
V5A_REAL_THREAT_ACCURACY = 0.789   # evaluation/v5a_seed999_results.json, n=493
V5A_REAL_PAIR_ACCURACY = 0.6356    # literal_pair_extraction.py on the same run, n=494
THREAT_REGRESSION_TOLERANCE_PP = 0.05
PAIR_REGRESSION_TOLERANCE_PP = 0.08  # widened -- see module docstring, bar i
FRESH_NULL_HYPOTHESIS_BASELINE = 0.0074  # evaluation/v5a2_null_hypothesis_baseline.json

BARS = {
    "threat_accuracy_pooled_when_answerable": {"op": ">=", "value": 0.55},
    "pair_accuracy_pooled_when_answerable_real": {"op": ">=", "value": 0.45},
    "ceiling_normalized_accuracy": {"op": ">=", "value": 0.85},
    "over_abstention_rate": {"op": "<=", "value": 0.15},
    "correct_abstention_rate_multi_hop": {"op": ">=", "value": 0.25},
    "correct_abstention_rate_oscillation": {"op": ">=", "value": 0.20},
    "escalation_under_escalation_rate": {"op": "<=", "value": 0.25},
    "regression_vs_v5a_threat_accuracy": {"op": ">=", "value": V5A_REAL_THREAT_ACCURACY - THREAT_REGRESSION_TOLERANCE_PP},
    "regression_vs_v5a_pair_accuracy": {"op": ">=", "value": V5A_REAL_PAIR_ACCURACY - PAIR_REGRESSION_TOLERANCE_PP},
}


def load_seed999_items():
    data = json.loads(SEED999_EVAL_SET.read_text())
    return data["items"]


def real_pair_accuracy(parsed_by_case: dict, items: list[dict]) -> float | None:
    gt_true = [it for it in items if it["has_ground_truth"]]
    n_correct = 0
    n_scored = 0
    for it in gt_true:
        parsed = parsed_by_case.get(it["name"])
        if parsed is None:
            continue
        n_scored += 1
        extracted = extract_literal_pair(parsed)
        true_pair = true_pair_from_chain(it["true_chain"])
        n_correct += extracted == true_pair
    return n_correct / n_scored if n_scored else None


def correct_abstention_and_per_mechanism(parsed_by_case: dict, items: list[dict]) -> dict:
    """Fraction of has_ground_truth=False cases correctly abstained on, pooled (diagnostic
    only -- see module docstring, "correct_abstention_rate_pooled" is mathematically the
    n-weighted average of the two per-mechanism rates below, not independent information) and
    split by ground-truth mechanism (Phase 3a simulator-based classifier on true_chain, NEVER
    stgt_bridge's read -- LOCKED CONFIG)."""
    gt_false = [it for it in items if not it["has_ground_truth"]]
    by_mechanism: dict[str, list[bool]] = {}
    pooled = []
    for it in gt_false:
        parsed = parsed_by_case.get(it["name"])
        if parsed is None:
            continue
        mechanism = classify_trajectory_ground_truth(it["true_chain"], true_labels=None)
        if mechanism is None:
            continue  # not actually unanswerable by this classifier -- skip, don't misscore
        correct = is_abstention(str(parsed.get("likely_intent", "")))
        pooled.append(correct)
        by_mechanism.setdefault(mechanism, []).append(correct)

    def rate(bools):
        return sum(bools) / len(bools) if bools else None

    return {
        "correct_abstention_rate_pooled": rate(pooled),
        "n_pooled": len(pooled),
        "correct_abstention_rate_multi_hop": rate(by_mechanism.get(MULTI_HOP, [])),
        "n_multi_hop": len(by_mechanism.get(MULTI_HOP, [])),
        "correct_abstention_rate_oscillation": rate(by_mechanism.get(OSCILLATION, [])),
        "n_oscillation": len(by_mechanism.get(OSCILLATION, [])),
    }


def extract_metrics(results: dict, memorization: dict | None, items: list[dict] | None = None) -> dict:
    answerability = results.get("answerability", {})
    escalation = results.get("escalation", {})
    parsed_by_case = results.get("parsed_by_case", {})

    if items is None:
        items = load_seed999_items()
    pair_acc = real_pair_accuracy(parsed_by_case, items) if parsed_by_case else None
    abst = correct_abstention_and_per_mechanism(parsed_by_case, items) if parsed_by_case else {}

    threat_acc = answerability.get("accuracy_when_answerable")
    ceiling_norm = (threat_acc / SAME_POPULATION_CEILING) if threat_acc is not None else None

    memo_rate = memorization.get("overlap_rate") if memorization else None

    return {
        "threat_accuracy_pooled_when_answerable": threat_acc,
        "pair_accuracy_pooled_when_answerable_real": pair_acc,
        "ceiling_normalized_accuracy": ceiling_norm,
        "over_abstention_rate": answerability.get("over_abstention_rate"),
        "correct_abstention_rate_pooled_diagnostic_only": abst.get("correct_abstention_rate_pooled"),
        "correct_abstention_rate_multi_hop": abst.get("correct_abstention_rate_multi_hop"),
        "correct_abstention_rate_oscillation": abst.get("correct_abstention_rate_oscillation"),
        "escalation_under_escalation_rate": escalation.get("under_escalated"),
        "escalation_over_escalation_rate": escalation.get("over_escalated"),
        "regression_vs_v5a_threat_accuracy": threat_acc,
        "regression_vs_v5a_pair_accuracy": pair_acc,
        "memorization_overlap_rate": memo_rate,
        "schema_validity_rate_diagnostic_only": results.get("schema_validity_rate", {}).get("rate"),
    }


def check_bar(value, bar: dict) -> str:
    if value is None:
        return "MISSING"
    if bar["op"] == ">=":
        return "PASS" if value >= bar["value"] else "FAIL"
    if bar["op"] == "<=":
        return "PASS" if value <= bar["value"] else "FAIL"
    raise ValueError(f"unknown op {bar['op']}")


def check_memorization(rate: float | None) -> str:
    if rate is None:
        return "MISSING"
    if rate >= 0.15:
        return f"FAIL -- MEMORIZATION SIGNAL ({rate:.1%} >= 15.0% bar)"
    return (f"PASS -- {rate:.1%} vs fresh null-hypothesis baseline "
            f"{FRESH_NULL_HYPOTHESIS_BASELINE:.1%}, below the 15.0% signal bar")


def run_check(results: dict, memorization: dict | None, items: list[dict] | None = None) -> tuple[list[dict], str]:
    metrics = extract_metrics(results, memorization, items=items)
    rows = []
    for name, bar in BARS.items():
        value = metrics[name]
        status = check_bar(value, bar)
        if name == "pair_accuracy_pooled_when_answerable_real" and status != "MISSING":
            status += " (REAL, literal -- see docs/PREREGISTRATION_V5A2.md section 1)"
        rows.append({
            "bar": name, "target": f"{bar['op']} {bar['value']:.1%}",
            "actual": f"{value:.1%}" if value is not None else "n/a", "status": status,
        })

    memo_status = check_memorization(metrics["memorization_overlap_rate"])
    rows.append({
        "bar": "memorization_vs_generalization", "target": "< 15.0% signal bar",
        "actual": (f"{metrics['memorization_overlap_rate']:.1%}"
                  if metrics["memorization_overlap_rate"] is not None else "n/a"),
        "status": memo_status,
    })

    pooled_abst = metrics["correct_abstention_rate_pooled_diagnostic_only"]
    rows.append({
        "bar": "correct_abstention_rate_pooled (DIAGNOSTIC ONLY, not scored -- mathematically "
               "the n-weighted average of the two mechanism bars above, see erratum)",
        "target": "n/a",
        "actual": f"{pooled_abst:.1%}" if pooled_abst is not None else "n/a", "status": "N/A",
    })

    schema_rate = metrics["schema_validity_rate_diagnostic_only"]
    rows.append({
        "bar": "schema_validity_rate (DIAGNOSTIC ONLY, not scored)", "target": "n/a",
        "actual": f"{schema_rate:.1%}" if schema_rate is not None else "n/a", "status": "N/A",
    })

    scored_rows = [r for r in rows if r["status"] not in ("N/A",)]
    overall = "PASS" if all(r["status"] == "PASS" or r["status"].startswith("PASS")
                           for r in scored_rows) else "FAIL"
    return rows, overall


def print_report(rows: list[dict], overall: str):
    print("| bar | target | actual | status |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['bar']} | {r['target']} | {r['actual']} | {r['status']} |")
    print(f"\nOVERALL: {overall}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", help="path to a v5a2 eval results JSON over evaluation/seed999_eval_set.json")
    ap.add_argument("--memorization", default=None,
                    help="path to a memorization overlap-rate results JSON "
                         "(score_memorization.py output, must include 'overlap_rate')")
    args = ap.parse_args()

    with open(args.results_json) as f:
        results = json.load(f)
    memorization = None
    if args.memorization:
        with open(args.memorization) as f:
            memorization = json.load(f)

    rows, overall = run_check(results, memorization)
    print_report(rows, overall)
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
