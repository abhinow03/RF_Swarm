"""
V5 program, Phase 0 (post strategy 5, docs/HISTORY.md): the metric HALT GATE 1 should
actually be judged against is not exact-pair accuracy. RULES (llm_finetuning/
build_sft_dataset.py) maps 49 (from, to) pairs onto only 4 threat levels, so a wrong
recovered pair often still yields the CORRECT threat (e.g. recovering
("v_shape","converging") when truth is ("column","converging") -- both RULES to "high").
Exact-pair accuracy is therefore a strictly pessimistic lower bound on the number that
actually gates end-to-end tactical accuracy: whether the downstream threat_level is right.

Re-scores the ALREADY-COMPUTED pair_records from phase0_ceiling.py's most recent run
(evaluation/phase0_ceiling_v5.json, strategy 5, 1000 trajectories, seed=999, 509
pair-eligible) against RULES for threat_level, likely_intent, and recommended_action.
No retraining, no new sampling, no model load -- pure re-scoring of existing recovered/
ground-truth pairs.

A record with no recovered pair (bucket B/C -- stgt_bridge couldn't resolve a clean (a,b))
counts as wrong for every metric here, same convention as phase0_ceiling.py's own
pair_correct field. This is the same "the system committed to no answer" case, scored
consistently across pair/threat/intent/action.

Usage:
    python scripts/phase0_threat_ceiling.py [--in evaluation/phase0_ceiling_v5.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "llm_finetuning"))

from build_sft_dataset import RULES  # noqa: E402

THREAT_LEVELS = ["low", "medium", "high", "critical"]


def score(records, pair_field, correct_field):
    n_eligible = len(records)
    n_pair_correct = sum(1 for r in records if r[correct_field])

    threat_conf = Counter()  # (true_threat, pred_threat_or_"no_recovery") -> count
    intent_correct = threat_correct = action_correct = 0
    n_no_recovery = 0

    for r in records:
        true_pair = tuple(r["gt_pair"])
        true_threat, true_intent, true_action = RULES[true_pair]
        rec = r[pair_field]
        if rec is None:
            n_no_recovery += 1
            threat_conf[(true_threat, "no_recovery")] += 1
            continue
        rec_pair = tuple(rec)
        rec_threat, rec_intent, rec_action = RULES[rec_pair]
        threat_conf[(true_threat, rec_threat)] += 1
        threat_correct += rec_threat == true_threat
        intent_correct += rec_intent == true_intent
        action_correct += rec_action == true_action

    return {
        "n_eligible": n_eligible,
        "n_no_recovery": n_no_recovery,
        "pair_accuracy": n_pair_correct / n_eligible,
        "threat_accuracy": threat_correct / n_eligible,
        "intent_accuracy": intent_correct / n_eligible,
        "action_accuracy": action_correct / n_eligible,
        "threat_confusion": threat_conf,
    }


def print_confusion(conf, title):
    cols = THREAT_LEVELS + ["no_recovery"]
    print(f"\n{title}")
    print("| true\\pred | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for t in THREAT_LEVELS:
        row = [str(conf.get((t, p), 0)) for p in cols]
        print(f"| {t} | " + " | ".join(row) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(REPO / "evaluation" / "phase0_ceiling_v5.json"))
    ap.add_argument("--out", default=str(REPO / "evaluation" / "phase0_threat_ceiling_v5.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text())
    records = data["pair_records"]
    assert len(RULES) == 49, f"expected 49 RULES entries, found {len(RULES)}"

    variants = [
        ("robust=False (shipped default)", "recovered_pair", "pair_correct"),
        ("robust=True (majority-vote reduction)", "recovered_pair_robust", "pair_correct_robust"),
    ]

    results = {}
    for label, pair_field, correct_field in variants:
        res = score(records, pair_field, correct_field)
        results[label] = res
        print("\n" + "=" * 100)
        print(label)
        print("=" * 100)
        print(f"n_eligible={res['n_eligible']}  n_no_recovery={res['n_no_recovery']}")
        print(f"(a) exact pair accuracy:   {res['pair_accuracy']:.1%}")
        print(f"(b) THREAT ceiling:        {res['threat_accuracy']:.1%}   <-- judge HALT GATE 1 against this")
        print(f"(c) intent ceiling:        {res['intent_accuracy']:.1%}")
        print(f"    action ceiling:        {res['action_accuracy']:.1%}")
        print_confusion(res["threat_confusion"], f"(d) THREAT confusion matrix (4 levels + no_recovery), {label}")

    out_path = Path(args.out)
    serializable = {
        label: {
            **{k: v for k, v in res.items() if k != "threat_confusion"},
            "threat_confusion": {f"{t}|{p}": c for (t, p), c in res["threat_confusion"].items()},
        }
        for label, res in results.items()
    }
    serializable["source"] = args.inp
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
