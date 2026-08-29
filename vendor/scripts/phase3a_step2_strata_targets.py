"""
Phase 3a step 2: propose preregistered per-mechanism row targets for the new abstention
corpus, using AUDIT.md sec AK's measured mechanism proportions (re-derived here via step 1's
ground-truth classifier, not sec AK's own STGT-noisy numbers) as the prior. Written BEFORE
any generation happens -- same discipline as Phase 1's STRATA_TARGETS
(llm_finetuning/build_sft_dataset.py) -- so the targets aren't quietly adjusted after seeing
what generation produces.

Reasoning:
  - multi_hop / oscillation: sec AK's 502-case population, RE-SCORED under step 1's
    ground-truth classifier (not the STGT-derived numbers, which sec AN proved noisy):
    435 multi_hop (86.7%) / 67 oscillation (13.3%) -- see
    evaluation/phase3a_ground_truth_validation.json. A total of 900 rows split at this exact
    ratio (780 multi_hop / 120 oscillation) is proposed: large enough to meaningfully teach
    the dominant real-world abstention behavior (sec AK: multi_hop/oscillation together are
    98.0% of ALL real unanswerable observations), small enough (~7.5% of the existing
    12,001-row corpus) not to skew the corpus toward over-abstention on answerable cases --
    PREREGISTRATION.md already flags over_abstention_rate<=15% as gameable at low abstention
    volume; a large abstention injection would risk the opposite failure mode.
  - terminal_transitioning: measured at 0/502 (0.0%) under the ground-truth classifier --
    it CANNOT occur naturally in the standard generation regime (every hop always completes
    its dwell period; see ground_truth_abstention.py's docstring). A strict frequency match
    would allocate 0 rows, which would leave the model with zero training signal for a real,
    distinct, structurally valid abstention mechism (an observer catching a trajectory
    mid-transition, one hop, is a completely ordinary real-world observation shape).
    Deliberately over-sampled relative to natural frequency, same precedent as Phase 1's own
    STRATA_TARGETS over-sampling "critical" threat-tier rows relative to its natural rarity.
    Proposed: 100 rows (~11% of multi_hop's count) -- enough for the model to reliably learn
    the pattern, not so many it dominates the mix.

Total: 780 + 120 + 100 = 1,000 new abstention rows (~8.3% of the existing 12,001-row corpus).

Usage:
    python scripts/phase3a_step2_strata_targets.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GT_VALIDATION_PATH = REPO / "evaluation" / "phase3a_ground_truth_validation.json"

TOTAL_MULTIHOP_OSCILLATION = 900
TERMINAL_TRANSITIONING_TARGET = 100

STRATA_TARGETS = {
    "multi_hop": 780,
    "oscillation": 120,
    "terminal_transitioning": TERMINAL_TRANSITIONING_TARGET,
}


def main():
    gt = json.loads(GT_VALIDATION_PATH.read_text())
    counts = gt["ground_truth_mechanism_counts"]
    n_mh, n_osc = counts["multi_hop"], counts["oscillation"]
    n_total = n_mh + n_osc
    ratio_mh, ratio_osc = n_mh / n_total, n_osc / n_total

    computed_mh = round(TOTAL_MULTIHOP_OSCILLATION * ratio_mh)
    computed_osc = TOTAL_MULTIHOP_OSCILLATION - computed_mh
    assert computed_mh == STRATA_TARGETS["multi_hop"], (
        f"documented target {STRATA_TARGETS['multi_hop']} does not match the ratio-derived "
        f"{computed_mh} -- targets must be re-derived, not hand-typed independently of the prior")
    assert computed_osc == STRATA_TARGETS["oscillation"]

    total = sum(STRATA_TARGETS.values())
    print("=== Phase 3a preregistered strata targets ===")
    print(f"prior (sec AK population, ground-truth-corrected): "
         f"multi_hop={n_mh}/{n_total} ({ratio_mh:.1%}), oscillation={n_osc}/{n_total} ({ratio_osc:.1%})")
    print(f"\nSTRATA_TARGETS: {STRATA_TARGETS}")
    print(f"total new abstention rows: {total}")
    print(f"as % of existing 12,001-row corpus: {100*total/12001:.1f}%")

    out = {"prior_multi_hop": n_mh, "prior_oscillation": n_osc, "prior_total": n_total,
          "prior_ratio_multi_hop": ratio_mh, "prior_ratio_oscillation": ratio_osc,
          "strata_targets": STRATA_TARGETS, "total_target": total,
          "pct_of_existing_corpus": 100 * total / 12001}
    out_path = REPO / "evaluation" / "phase3a_strata_targets.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
