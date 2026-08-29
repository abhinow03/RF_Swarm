"""Carve a fresh stratified train/val split for v5a2's merged corpus.

data/sft_train_v5_phase3a_merged.jsonl (12,901 rows) is a flat pool of
v5-a's phase1 train (10,801) + phase1 val (600) + phase1 mining (600) +
the new abstention_900 corpus (multi_hop=780, oscillation=120), concatenated
in that exact order (verified against the source files). It has no held-out
val split of its own -- the phase3a merge (sec AP) preserved row count and
dedup but not the train/val partition. This script carves one.

Stratification:
  - Non-abstention rows (first 12,001) are stratified by threat_level
    (low/medium/high/critical), parsed from the assistant message JSON.
    This replicates Phase 1's own val construction: phase1 train/val/mining
    all have closely matching threat_level proportions (see AUDIT.md sec AQ),
    i.e. Phase 1's val was itself a proportional stratified sample of the
    threat_level distribution -- so proportional stratified sampling here is
    a faithful replication, not a new design choice.
  - Abstention rows (last 900) are stratified by mechanism (multi_hop=780,
    oscillation=120), taken from abstention_corpus_teacher_trimmed900_meta.json's
    rows_detail, whose order was verified to match the jsonl row order.

Split fraction: 600/10801 (v5-a's original val fraction), applied per-stratum
so overall val share of the corpus matches v5-a's, without any stratum being
absent or over/under-represented relative to its own share of the pool.

Seeded (SEED=20260816) with Python's random.Random for reproducibility.
"""
import json
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MERGED = REPO / "data" / "sft_train_v5_phase3a_merged.jsonl"
ABST_META = REPO / "data" / "abstention_corpus_teacher_trimmed900_meta.json"
OUT_TRAIN = REPO / "data" / "sft_train_v5a2_train.jsonl"
OUT_VAL = REPO / "data" / "sft_train_v5a2_val.jsonl"

N_NON_ABSTENTION = 12001  # phase1 train(10801) + val(600) + mining(600)
N_ABSTENTION = 900
SPLIT_FRACTION = 600 / 10801
SEED = 20260816
MIN_STRATUM_FOR_SPLIT = 20  # below this, flag rather than silently split


def threat_level(line: str) -> str:
    d = json.loads(line)
    content = d["messages"][1]["content"]
    j = json.loads(content)
    return j.get("threat_level", "MISSING")


def main() -> None:
    lines = MERGED.read_text().splitlines()
    assert len(lines) == N_NON_ABSTENTION + N_ABSTENTION, len(lines)

    meta = json.loads(ABST_META.read_text())
    mechanisms = [d["mechanism"] for d in meta["rows_detail"]]
    assert len(mechanisms) == N_ABSTENTION
    assert Counter(mechanisms) == Counter({"multi_hop": 780, "oscillation": 120})

    # assign a stratum label to every row by global index
    strata: dict[str, list[int]] = defaultdict(list)
    for i, line in enumerate(lines):
        if i < N_NON_ABSTENTION:
            label = threat_level(line)
            assert label in ("low", "medium", "high", "critical"), (i, label)
        else:
            label = mechanisms[i - N_NON_ABSTENTION]
        strata[label].append(i)

    rng = random.Random(SEED)
    val_indices: set[int] = set()
    breakdown = {}
    for label, idxs in sorted(strata.items()):
        n_stratum = len(idxs)
        n_val = round(n_stratum * SPLIT_FRACTION)
        flagged = n_stratum < MIN_STRATUM_FOR_SPLIT
        sampled = set(rng.sample(idxs, n_val)) if n_val > 0 else set()
        val_indices |= sampled
        breakdown[label] = {
            "pool_rows": n_stratum,
            "val_rows": len(sampled),
            "train_rows": n_stratum - len(sampled),
            "val_fraction_actual": len(sampled) / n_stratum if n_stratum else 0.0,
            "flagged_small_stratum": flagged,
        }

    train_lines = [line for i, line in enumerate(lines) if i not in val_indices]
    val_lines = [line for i, line in enumerate(lines) if i in val_indices]

    assert len(train_lines) + len(val_lines) == len(lines)

    # zero-overlap check: actually diff content hashes, not just index bookkeeping
    train_hashes = {hashlib.sha256(l.encode()).hexdigest() for l in train_lines}
    val_hashes = {hashlib.sha256(l.encode()).hexdigest() for l in val_lines}
    overlap = train_hashes & val_hashes
    # NOTE: this checks whole-row content hashes. If any duplicate row content
    # existed pre-split (it shouldn't -- phase3a merge deduped), a row could
    # land content-identical in both files despite disjoint indices. Report
    # explicitly either way.

    OUT_TRAIN.write_text("\n".join(train_lines) + "\n")
    OUT_VAL.write_text("\n".join(val_lines) + "\n")

    train_sha = hashlib.sha256(OUT_TRAIN.read_bytes()).hexdigest()
    val_sha = hashlib.sha256(OUT_VAL.read_bytes()).hexdigest()

    print("=== v5a2 train/val carve report ===")
    print(f"seed: {SEED}")
    print(f"split_fraction (v5-a's val fraction, 600/10801): {SPLIT_FRACTION:.6f}")
    print()
    print(f"{'stratum':<12} {'pool':>6} {'val':>6} {'train':>6} {'val_frac':>10} {'flagged':>8}")
    for label, b in sorted(breakdown.items()):
        print(
            f"{label:<12} {b['pool_rows']:>6} {b['val_rows']:>6} {b['train_rows']:>6} "
            f"{b['val_fraction_actual']:>10.4f} {str(b['flagged_small_stratum']):>8}"
        )
    print()
    print(f"total rows: {len(lines)}")
    print(f"train rows: {len(train_lines)}  ({len(train_lines)/len(lines)*100:.2f}%)")
    print(f"val rows:   {len(val_lines)}  ({len(val_lines)/len(lines)*100:.2f}%)")
    print(f"train + val == 12901: {len(train_lines) + len(val_lines) == 12901}")
    print()
    print(f"zero row-level overlap (content-hash diff): {len(overlap) == 0} (overlap count={len(overlap)})")
    print()
    print(f"{OUT_TRAIN} sha256: {train_sha}")
    print(f"{OUT_VAL} sha256:   {val_sha}")


if __name__ == "__main__":
    main()
