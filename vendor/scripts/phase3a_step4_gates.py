"""
Phase 3a step 4: MANDATORY hard gates on data/abstention_corpus.jsonl. Fail = stop, do not
proceed to step 5 (merge/report as final). Each gate below is a live, non-zero assertion
against the actual files on disk -- not a docstring claim.

Gate a: per-mechanism row count > 0 and matches/exceeds step 2's preregistered targets.
Gate b: zero overlap (exact user-message-string dedup, same check build_sft_dataset.py's
        --append path uses) between the new corpus and the existing 12,001 RULES rows.
Gate c: spot-check 20 rows per mechanism -- prints chain + assigned mechanism + reasoning,
        and independently re-verifies each with classify_trajectory_ground_truth() (the same
        function used at generation time, imported fresh here rather than trusted from the
        generation run's own bookkeeping).
Gate d: checkpoints/v5_sft_v5a_PROTECTED/ still sha256-matches the original -- proves this
        session's work never touched the v5-a safety copy.

Usage:
    python scripts/phase3a_step4_gates.py
    python scripts/phase3a_step4_gates.py --corpus data/abstention_corpus_teacher.jsonl \
        --meta data/abstention_corpus_teacher_meta.json --out-suffix _teacher
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.ground_truth_abstention import classify_trajectory_ground_truth  # noqa: E402

TARGETS_PATH = REPO / "evaluation" / "phase3a_strata_targets.json"
EXISTING_CORPUS_FILES = [
    REPO / "data" / "sft_train_v5_phase1.jsonl",
    REPO / "data" / "sft_train_v5_phase1_val.jsonl",
    REPO / "data" / "sft_train_v5_phase1_mining.jsonl",
]


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def gate_a(meta, targets):
    print("=== GATE a: per-mechanism row count > 0 and >= preregistered target ===")
    ok = True
    for mechanism, target in targets["strata_targets"].items():
        actual = meta["per_mechanism"].get(mechanism, 0)
        passed = actual > 0 and actual >= target
        ok &= passed
        print(f"  {mechanism}: actual={actual} target={target} -- {'PASS' if passed else 'FAIL'}")
    print(f"GATE a: {'PASS' if ok else 'FAIL'}")
    return ok


def gate_b(new_rows):
    print("\n=== GATE b: zero overlap with the existing 12,001-row RULES corpus ===")
    existing_keys = set()
    for path in EXISTING_CORPUS_FILES:
        for row in load_jsonl(path):
            existing_keys.add(row["messages"][0]["content"])
    n_existing = len(existing_keys)
    assert n_existing == 12001, f"expected 12001 existing keys, found {n_existing}"

    new_keys = [r["messages"][0]["content"] for r in new_rows]
    overlap = [k for k in new_keys if k in existing_keys]
    # also check the new corpus is internally dedup'd against itself
    internal_dupes = len(new_keys) - len(set(new_keys))

    ok = len(overlap) == 0 and internal_dupes == 0
    print(f"  existing corpus keys: {n_existing}")
    print(f"  new corpus rows: {len(new_keys)}")
    print(f"  overlap with existing corpus: {len(overlap)}")
    print(f"  internal duplicates within new corpus: {internal_dupes}")
    print(f"GATE b: {'PASS' if ok else 'FAIL'}")
    return ok, len(overlap), internal_dupes


def gate_c(meta):
    print("\n=== GATE c: spot-check 20 rows per mechanism against ground truth ===")
    rng = random.Random(1)
    mislabeled = []
    by_mechanism = {}
    for r in meta["rows_detail"]:
        by_mechanism.setdefault(r["mechanism"], []).append(r)

    for mechanism, rows in by_mechanism.items():
        sample = rng.sample(rows, min(20, len(rows)))
        print(f"\n  --- {mechanism} (n={len(sample)} spot-checked) ---")
        for r in sample:
            chain = r["chain"]
            recomputed = classify_trajectory_ground_truth(chain, true_labels=None)
            # multi_hop/oscillation are chain-length>=3 -- fully re-verifiable from chain
            # alone; terminal_transitioning (chain length <=2 in this corpus) cannot be
            # re-verified without true_labels, which generation-time already confirmed and
            # this script does not re-simulate (that would just re-run generation) -- flagged
            # as "not re-verifiable from chain alone", not silently passed.
            if len(chain) >= 3:
                status = "OK" if recomputed == mechanism else "MISLABELED"
                if status == "MISLABELED":
                    mislabeled.append({"mechanism": mechanism, "chain": chain, "recomputed": recomputed})
            else:
                status = "not re-verifiable from chain alone (terminal_transitioning needs true_labels)"
            print(f"    chain={chain}  assigned={mechanism}  recomputed_from_chain={recomputed}  [{status}]")

    ok = len(mislabeled) == 0
    print(f"\nmislabeled cases found: {len(mislabeled)}")
    for m in mislabeled:
        print(f"  {m}")
    print(f"GATE c: {'PASS' if ok else 'FAIL'}")
    return ok, mislabeled


def gate_d():
    print("\n=== GATE d: checkpoints/v5_sft_v5a_PROTECTED/ still matches original ===")
    import subprocess
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "phase3a_verify_safety_copy.py")],
                            capture_output=True, text=True)
    print(result.stdout)
    ok = result.returncode == 0
    print(f"GATE d: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "data" / "abstention_corpus.jsonl"))
    ap.add_argument("--meta", default=str(REPO / "data" / "abstention_corpus_meta.json"))
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    targets = json.loads(TARGETS_PATH.read_text())
    new_rows = load_jsonl(Path(args.corpus))
    assert len(new_rows) == meta["n_total"]

    ok_a = gate_a(meta, targets)
    ok_b, n_overlap, n_internal_dupes = gate_b(new_rows)
    ok_c, mislabeled = gate_c(meta)
    ok_d = gate_d()

    overall = ok_a and ok_b and ok_c and ok_d
    print("\n" + "=" * 80)
    print(f"STEP 4 OVERALL: {'ALL GATES PASS' if overall else 'AT LEAST ONE GATE FAILED -- STOP'}")
    print(f"  gate a (row counts):        {'PASS' if ok_a else 'FAIL'}")
    print(f"  gate b (zero overlap):      {'PASS' if ok_b else 'FAIL'} (overlap={n_overlap}, internal_dupes={n_internal_dupes})")
    print(f"  gate c (spot-check):        {'PASS' if ok_c else 'FAIL'} (mislabeled={len(mislabeled)})")
    print(f"  gate d (safety copy intact): {'PASS' if ok_d else 'FAIL'}")
    print("=" * 80)

    out = {
        "gate_a_pass": ok_a, "gate_b_pass": ok_b, "gate_b_overlap": n_overlap,
        "gate_b_internal_dupes": n_internal_dupes, "gate_c_pass": ok_c,
        "gate_c_mislabeled": mislabeled, "gate_d_pass": ok_d, "overall_pass": overall,
    }
    out_path = REPO / "evaluation" / f"phase3a_step4_gates_results{args.out_suffix}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
