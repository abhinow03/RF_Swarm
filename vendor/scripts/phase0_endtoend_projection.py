"""
V5 program, Phase 0, post-threshold-sweep step 6 (docs/HISTORY.md): HALT GATE 1's
70% floor was set when pair-level and threat-level ceilings were nearly identical
(12.2% vs 13.0%, strategy 5, pre-guard-fix) -- a floor on ONE number implicitly
gated both. They have since diverged sharply (47.0% vs 52.3%, robust=False,
post-guard-fix). This computes the projected END-TO-END threat accuracy the
current pipeline implies, decomposed exactly as requested:

    end_to_end = current_measured_threat_ceiling + P(bucket C) * layer_3_accuracy_estimate

This identity holds because `current_measured_threat_ceiling` (scripts/
phase0_threat_ceiling.py's output) is ALREADY `P(bucket A) * threat_accuracy_within_A`
-- bucket B/C both score 0 in that metric (no LLM layer involved, "no recovery =
wrong"). Adding Layer 3's real contribution on bucket C is the only missing term;
bucket B (guard) contributes 0 either way, by design (Layer 2 abstains).

Pure post-processing of two files already on disk (`evaluation/
phase0_ceiling_v5_guardfix.json`'s `pair_records`, `evaluation/
phase0_threat_ceiling_v5_guardfix.json`) plus ONE external, disclosed number:
v3b-fix's measured accuracy on real STGT output from the earlier engagement
(AUDIT.md sec AF, `evaluation/eval_real_stgt_output.json`) -- a different
checkpoint/bridge state (pre-guard-fix), not bucket-conditioned (that eval ran
v3b-fix on EVERY case, not just bucket-C ones), so used as a DISCLOSED estimate
with an explicit sensitivity range, not re-measured this turn (no training, and
a fresh LLM eval is out of scope for a same-day projection turn).

Usage:
    python scripts/phase0_endtoend_projection.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "llm_finetuning"))

from build_sft_dataset import RULES  # noqa: E402

# From AUDIT.md sec AF (earlier engagement, pre-guard-fix checkpoint/bridge):
# v3b-fix's overall threat accuracy on the 249 ground-truth-determinable real
# sequences, unconditioned on bucket (it answered every case, not just the
# bucket-C-shaped ones Layer 3 would actually see) -- evaluation/
# eval_real_stgt_output.json: correct=26/... no, v3b-fix specifically: 77/249.
V3B_FIX_SEC_AF_ACCURACY = 77 / 249  # 30.9%
SENSITIVITY_RANGE = (0.20, V3B_FIX_SEC_AF_ACCURACY, 0.40)  # conservative, central, optimistic


def threat_correct(pair_a, pair_b):
    if pair_a is None or pair_b is None:
        return False
    if tuple(pair_a) not in RULES or tuple(pair_b) not in RULES:
        return False
    return RULES[tuple(pair_a)][0] == RULES[tuple(pair_b)][0]


def main():
    ceiling = json.loads((REPO / "evaluation" / "phase0_ceiling_v5_guardfix.json").read_text())
    threat = json.loads((REPO / "evaluation" / "phase0_threat_ceiling_v5_guardfix.json").read_text())
    records = ceiling["pair_records"]
    n = len(records)

    print(f"=== end-to-end threat accuracy projection (n_pair_eligible={n}) ===\n")

    for mode, bucket_key, recovered_key, label in (
        ("robust_false", "bucket", "recovered_pair", "robust=False (current shipped default)"),
        ("robust_true", "bucket_robust", "recovered_pair_robust", "robust=True @ threshold=0.7"),
    ):
        bucket_a_records = [r for r in records if r[bucket_key] == "A"]
        bucket_c_records = [r for r in records if r[bucket_key] == "C"]
        bucket_b_records = [r for r in records if r[bucket_key] == "B"]
        n_a, n_b, n_c = len(bucket_a_records), len(bucket_b_records), len(bucket_c_records)

        threat_hits_a = sum(1 for r in bucket_a_records if threat_correct(r["gt_pair"], r[recovered_key]))
        threat_acc_within_a = threat_hits_a / n_a if n_a else 0.0

        current_ceiling_key = ("robust=False (shipped default)" if mode == "robust_false"
                               else "robust=True (majority-vote reduction)")
        current_measured_ceiling = threat[current_ceiling_key]["threat_accuracy"]

        # identity check: current_measured_ceiling should equal (n_a * threat_acc_within_a) / n
        identity_check = (n_a * threat_acc_within_a) / n

        print(f"--- {label} ---")
        print(f"  bucket distribution: A={n_a} ({n_a/n:.1%}), B={n_b} ({n_b/n:.1%}), C={n_c} ({n_c/n:.1%})")
        print(f"  threat accuracy WITHIN bucket A: {threat_hits_a}/{n_a} = {threat_acc_within_a:.1%}")
        print(f"  current measured threat ceiling (phase0_threat_ceiling.py): {current_measured_ceiling:.1%}")
        print(f"  identity check (n_A * acc|A / n): {identity_check:.1%} "
             f"({'MATCH' if abs(identity_check - current_measured_ceiling) < 0.005 else 'MISMATCH -- investigate'})")

        print(f"\n  end-to-end projection = current_ceiling + P(bucket C) * layer_3_accuracy_estimate:")
        p_c = n_c / n
        for label_s, layer3_acc in zip(("conservative", "central (sec AF v3b-fix)", "optimistic"),
                                       SENSITIVITY_RANGE):
            end_to_end = current_measured_ceiling + p_c * layer3_acc
            print(f"    Layer 3 accuracy = {layer3_acc:.1%} ({label_s}): "
                 f"end-to-end = {current_measured_ceiling:.1%} + {p_c:.1%}*{layer3_acc:.1%} "
                 f"= {end_to_end:.1%}")
        print()

    print("=== HALT GATE 1 re-examination ===")
    print("Original 70% floor was set on pair-level accuracy when pair-level (12.2%) and "
         "threat-level (13.0%) ceilings were nearly identical -- a single number implicitly "
         "gated both. They have since diverged to 47.0% (pair) vs 52.3% (threat, robust=False) "
         "-- a 5.3-point gap that will only widen as pair-level brittleness (chain-length-2, "
         "guard precision) is fixed without necessarily fixing threat-level in lockstep, since "
         "RULES' 49-pair-to-4-threat-level mapping means many wrong pairs still land on the "
         "right threat. A single 70% floor stated in pair-level terms is now measuring the "
         "wrong thing for a gate whose actual purpose is end-to-end TACTICAL correctness.")


if __name__ == "__main__":
    main()
