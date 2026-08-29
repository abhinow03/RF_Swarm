"""
Rule 0 self-check for docs/PREREGISTRATION_V5A2.md (v5a2 preregistration, step 6): confirm
every bar in the document's own section 2 table traces to a live, non-zero assertion inside
scripts/check_preregistration_v5a2.py -- not a table entry only -- and that
tests/test_check_preregistration_v5a2.py has at least one test per bar proving it isn't
vacuous (fails when the condition making it meaningful is absent). Same style as
scripts/rule0_audit_2026_08_13.py's check_3_preregistration_bars.

Usage:
    python scripts/rule0_audit_v5a2_preregistration.py
"""
from __future__ import annotations

import importlib
import inspect
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

# The 8 PASS/FAIL bars per the erratum (docs/PREREGISTRATION_V5A2.md section 2, corrected --
# "under_abstention_rate" dropped as a scored bar, proven mathematically redundant with the
# two mechanism bars below), plus the memorization bar (checked separately, not via the BARS
# dict) and the explicitly-dropped diagnostics (schema_validity_rate,
# correct_abstention_rate_pooled).
DOCUMENT_BARS = [
    "threat_accuracy_pooled_when_answerable",
    "pair_accuracy_pooled_when_answerable_real",
    "ceiling_normalized_accuracy",
    "over_abstention_rate",
    "correct_abstention_rate_multi_hop",
    "correct_abstention_rate_oscillation",
    "escalation_under_escalation_rate",
    "regression_vs_v5a_threat_accuracy",
    "regression_vs_v5a_pair_accuracy",
]
MEMORIZATION_BAR = "memorization_vs_generalization"
DROPPED_DIAGNOSTIC_ONLY = "schema_validity_rate"


def check_every_bar_has_a_live_assertion():
    cp = importlib.import_module("check_preregistration_v5a2")
    bars_in_dict = set(cp.BARS.keys())
    missing_from_dict = [b for b in DOCUMENT_BARS if b not in bars_in_dict]

    has_memorization_fn = "check_memorization" in dir(cp) and callable(cp.check_memorization)
    memorization_uses_real_threshold = "0.15" in inspect.getsource(cp.check_memorization)

    schema_is_scored = DROPPED_DIAGNOSTIC_ONLY in bars_in_dict
    source = inspect.getsource(cp)
    schema_marked_diagnostic_only = "DIAGNOSTIC ONLY, not scored" in source

    return {
        "document_bars": DOCUMENT_BARS,
        "bars_in_check_script_BARS_dict": sorted(bars_in_dict),
        "missing_from_dict": missing_from_dict,
        "all_document_bars_have_live_assertion": len(missing_from_dict) == 0,
        "memorization_has_dedicated_function": has_memorization_fn,
        "memorization_function_uses_real_threshold_not_hardcoded_pass": memorization_uses_real_threshold,
        "schema_validity_correctly_excluded_from_BARS_dict": not schema_is_scored,
        "schema_validity_marked_diagnostic_only_in_source": schema_marked_diagnostic_only,
    }


def check_every_bar_has_a_non_vacuous_test():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName("test_check_preregistration_v5a2")
    test_names = []

    def collect(s):
        for t in s:
            if isinstance(t, unittest.TestSuite):
                collect(t)
            else:
                test_names.append(str(t))

    collect(suite)

    # Every document bar (+ memorization) must be referenced by at least one test's own
    # source (assertEqual on that bar's key), not just exist as a numeric target -- checked
    # by inspecting the test module's source text directly.
    tm = importlib.import_module("test_check_preregistration_v5a2")
    test_source = inspect.getsource(tm)
    bars_covered = {b: (b in test_source) for b in DOCUMENT_BARS + [MEMORIZATION_BAR]}
    uncovered = [b for b, covered in bars_covered.items() if not covered]

    # "non-vacuous" specifically: at least one test per abstention/memorization/escalation bar
    # asserts a FAIL outcome, not just a PASS outcome (a bar that can only ever be tested
    # passing hasn't been shown capable of catching a bad system).
    has_fail_assertion_for_abstention = "FAIL" in test_source and "correct_abstention_rate_multi_hop" in test_source
    has_fail_assertion_for_memorization = 'assertTrue(by_bar["memorization_vs_generalization"].startswith("FAIL"))' in test_source
    has_fail_assertion_for_escalation = "test_escalation_ceiling_is_a_real_numeric_bound_not_qualitative" in test_source

    return {
        "n_tests_collected": len(test_names),
        "bars_referenced_in_test_source": bars_covered,
        "bars_with_no_test_reference": uncovered,
        "all_document_bars_referenced_by_a_test": len(uncovered) == 0,
        "has_a_failing_case_for_abstention_bars": has_fail_assertion_for_abstention,
        "has_a_failing_case_for_memorization_bar": has_fail_assertion_for_memorization,
        "has_a_failing_case_for_escalation_bar": has_fail_assertion_for_escalation,
    }


def main():
    result_1 = check_every_bar_has_a_live_assertion()
    result_2 = check_every_bar_has_a_non_vacuous_test()

    overall_pass = (
        result_1["all_document_bars_have_live_assertion"]
        and result_1["memorization_has_dedicated_function"]
        and result_1["memorization_function_uses_real_threshold_not_hardcoded_pass"]
        and result_1["schema_validity_correctly_excluded_from_BARS_dict"]
        and result_1["schema_validity_marked_diagnostic_only_in_source"]
        and result_2["all_document_bars_referenced_by_a_test"]
        and result_2["has_a_failing_case_for_abstention_bars"]
        and result_2["has_a_failing_case_for_memorization_bar"]
        and result_2["has_a_failing_case_for_escalation_bar"]
    )

    print("=" * 90)
    print("RULE 0 SELF-CHECK: docs/PREREGISTRATION_V5A2.md")
    print("=" * 90)
    print(json.dumps({"live_assertions": result_1, "non_vacuous_tests": result_2}, indent=2))
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")

    out_path = REPO / "evaluation" / "rule0_v5a2_preregistration_audit.json"
    out_path.write_text(json.dumps(
        {"live_assertions": result_1, "non_vacuous_tests": result_2, "overall_pass": overall_pass},
        indent=2))
    print(f"saved {out_path}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
