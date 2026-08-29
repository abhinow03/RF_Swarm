"""
Phase 3a step 5: consolidated final-corpus report. Pulls every number from the actual
persisted artifacts (never re-typed) -- data/abstention_corpus.jsonl,
data/abstention_corpus_meta.json, evaluation/phase3a_step4_gates_results.json, and the
existing 12,001-row corpus files. data/abstention_corpus.jsonl is NOT merged into any
training file by this script or any other in this phase.

Usage:
    python scripts/phase3a_step5_final_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXISTING_CORPUS_FILES = [
    REPO / "data" / "sft_train_v5_phase1.jsonl",
    REPO / "data" / "sft_train_v5_phase1_val.jsonl",
    REPO / "data" / "sft_train_v5_phase1_mining.jsonl",
]


def count_lines(path):
    return sum(1 for l in path.read_text().splitlines() if l.strip())


def main():
    meta = json.loads((REPO / "data" / "abstention_corpus_meta.json").read_text())
    gates = json.loads((REPO / "evaluation" / "phase3a_step4_gates_results.json").read_text())
    n_new = count_lines(REPO / "data" / "abstention_corpus.jsonl")
    n_existing = sum(count_lines(p) for p in EXISTING_CORPUS_FILES)

    report = {
        "new_abstention_corpus_file": "data/abstention_corpus.jsonl",
        "merged_into_training_file": False,
        "n_new_rows_total": n_new,
        "per_mechanism_counts": meta["per_mechanism"],
        "teacher_used": meta["teacher_enabled"],
        "n_rows_with_teacher_prose": meta["n_used_teacher"],
        "existing_rules_corpus_row_count": n_existing,
        "combined_corpus_size_if_merged": n_existing + n_new,
        "dedup_confirmation": {
            "overlap_with_existing_corpus": gates["gate_b_overlap"],
            "internal_duplicates": gates["gate_b_internal_dupes"],
        },
        "hard_gates_all_pass": gates["overall_pass"],
        "safety_copy_verified_untouched": gates["gate_d_pass"],
    }

    assert n_new == meta["n_total"] == sum(meta["per_mechanism"].values())
    assert n_existing == 12001

    print(json.dumps(report, indent=2))
    out_path = REPO / "evaluation" / "phase3a_final_corpus_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
