"""
V5 program, Phase 0, pre-strategy-6 step 2 (docs/HISTORY.md): decompose WHY pair-recovery
fails. Window-level accuracy is 70.4% but pair-level accuracy is 12.2% -- roughly 0.704^7,
i.e. reduction behaves as though it needs ~7 consecutive correct windows. This script asks,
for every FAILED pair-eligible trajectory: how many windows were misclassified, at which
positions (leading/interior/trailing thirds of the window sequence), and -- the key
question -- would the correct pair still have been recoverable by majority-vote reduction
using ONLY the windows the classifier got right?

Re-runs inference (NOT training) with the exact same seed=999 sampling regime as
scripts/phase0_ceiling.py, so it reproduces the identical 1000 trajectories index-for-index.
For each of the 509 pair-eligible trajectories, filters the prediction list down to windows
where pred_label == true_label, then re-applies swarm_intent.coverage.classify_observation
to that filtered subset. If the correct pair IS recoverable from the correct-only subset,
that trajectory's failure is attributable to a few bad windows tripping the unanimity/guard
logic, not a broken underlying signal -- the reduction/guard pipeline, not the classifier,
is the bottleneck for that case.

Usage (run inside tmux -- ~1000 model inferences, several minutes):
    python scripts/phase0_decompose_failures.py --n 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.config import BASE_FORMATIONS, TRANSITION_CLASS  # noqa: E402
from swarm_intent.coverage import classify_observation, BUCKET_A  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402
# 2026-08-09 step 26 (docs/V5_LOG.md): consolidated here from an inline, independently-
# maintained copy -- see src/swarm_intent/eval_trajectories.py for the full derivation of
# both the destination-dwell (step 24/25) and source-lead-in (step 26) observability fixes.
# Re-exported under these same names so phase0_chainlength_breakdown.py's/
# phase0_chain2_observability.py's `from phase0_decompose_failures import (sample_chain,
# build_long_sequence_labeled, ground_truth_pair, ...)` keeps working unchanged.
from swarm_intent.eval_trajectories import (  # noqa: E402, F401
    sample_chain, build_long_sequence_labeled, ground_truth_pair,
)

DATA_DIR = REPO / "swarm_data"
CHECKPOINT = DATA_DIR / "best_model.pt"
SEED = 999
CLASS_ORDER = list(BASE_FORMATIONS) + [TRANSITION_CLASS]


def position_bucket(idx: int, n: int) -> str:
    if n <= 2:
        return "interior"
    if idx < n / 3:
        return "leading"
    if idx >= 2 * n / 3:
        return "trailing"
    return "interior"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(REPO / "evaluation" / "phase0_decompose_failures.json"))
    args = ap.parse_args()

    import torch
    from swarm_intent.stgt.config import device
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = STGTModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    rng = np.random.default_rng(args.seed)
    reporter = Reporter("phase0_decompose_failures", args.n, rate_hint=2.0)

    decompositions = []

    for i in range(args.n):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq, true_labels = build_long_sequence_labeled(chain, rng, spread, noise_std)

        gt_pair = ground_truth_pair(chain)
        if gt_pair is None:
            reporter.update(1, item=f"seq {i} (skip, chain>2)")
            continue

        predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)

        window_size, stride = 50, 10
        window_true, window_pred, correct_mask = [], [], []
        for pred in predictions:
            start = len(window_true) * stride
            end = min(start + window_size, len(true_labels))
            window_true_labels = true_labels[start:end]
            if not window_true_labels:
                window_true.append(None)
                window_pred.append(None)
                correct_mask.append(False)
                continue
            true_label = Counter(window_true_labels).most_common(1)[0][0]
            pred_label = pred["formation_type"] if pred["formation_type"] in CLASS_ORDER else TRANSITION_CLASS
            window_true.append(true_label)
            window_pred.append(pred_label)
            correct_mask.append(pred_label == true_label)

        n_windows = len(predictions)
        n_bad = sum(1 for c in correct_mask if not c)
        bad_positions = [position_bucket(idx, n_windows) for idx, c in enumerate(correct_mask) if not c]
        bad_position_counts = dict(Counter(bad_positions))

        bucket_info = classify_observation(predictions, robust=False)
        recovered = tuple(bucket_info["rules_key"]) if bucket_info["bucket"] == BUCKET_A else None
        pair_correct = recovered == gt_pair

        recoverable_from_correct_only = None
        if not pair_correct:
            filtered_predictions = [p for p, c in zip(predictions, correct_mask) if c]
            if filtered_predictions:
                filt_info = classify_observation(filtered_predictions, robust=False)
                filt_recovered = tuple(filt_info["rules_key"]) if filt_info["bucket"] == BUCKET_A else None
                recoverable_from_correct_only = (filt_recovered == gt_pair)
            else:
                recoverable_from_correct_only = False

        decompositions.append({
            "i": i, "true_chain": chain, "gt_pair": list(gt_pair),
            "chain_length": len(chain),
            "n_windows": n_windows, "n_bad_windows": n_bad,
            "bad_window_positions": bad_position_counts,
            "bucket": bucket_info["bucket"],
            "guard_reasons": bucket_info["guard_reasons"],
            "recovered_pair": list(recovered) if recovered else None,
            "pair_correct": pair_correct,
            "recoverable_from_correct_only_windows": recoverable_from_correct_only,
        })

        reporter.update(1, item=f"seq {i}")

    reporter.status = "done"
    reporter._write()

    failures = [d for d in decompositions if not d["pair_correct"]]
    n_total = len(decompositions)
    n_fail = len(failures)

    print("\n" + "=" * 100)
    print(f"PAIR-RECOVERY FAILURE DECOMPOSITION -- n_eligible={n_total}, n_failed={n_fail} "
         f"({n_fail / n_total:.1%})")
    print("=" * 100)

    print("\nBY CHAIN LENGTH (steady-state=1 vs. single-transition=2):")
    print("| chain_length | n | n_correct | accuracy |")
    print("|---|---|---|---|")
    for cl in (1, 2):
        sub = [d for d in decompositions if d["chain_length"] == cl]
        n_sub = len(sub)
        n_corr = sum(1 for d in sub if d["pair_correct"])
        print(f"| {cl} | {n_sub} | {n_corr} | {n_corr / n_sub:.1%} |" if n_sub else f"| {cl} | 0 | 0 | n/a |")

    print("\nGUARD REASONS among failures with ZERO bad windows (classifier was PERFECT on every "
         "window, still failed to recover the pair):")
    zero_bad_fail = [d for d in failures if d["n_bad_windows"] == 0]
    reason_counts = Counter()
    for d in zero_bad_fail:
        if d["guard_reasons"]:
            for r in d["guard_reasons"]:
                reason_counts[r] += 1
        else:
            reason_counts[f"bucket_{d['bucket']}_no_guard_reason"] += 1
    print(f"n zero-bad-window failures: {len(zero_bad_fail)}/{n_fail}")
    for reason, c in reason_counts.most_common():
        print(f"  {reason}: {c}")

    bad_window_hist = Counter(d["n_bad_windows"] for d in failures)
    print("\ndistribution of n_bad_windows among FAILED trajectories:")
    print("| n_bad_windows | count | fraction of failures |")
    print("|---|---|---|")
    for k in sorted(bad_window_hist):
        c = bad_window_hist[k]
        print(f"| {k} | {c} | {c / n_fail:.1%} |")

    n_1_2_bad = sum(c for k, c in bad_window_hist.items() if k in (1, 2))
    print(f"\nfailures with only 1-2 bad windows: {n_1_2_bad}/{n_fail} = {n_1_2_bad / n_fail:.1%}")

    pos_totals = Counter()
    for d in failures:
        for pos, c in d["bad_window_positions"].items():
            pos_totals[pos] += c
    total_bad = sum(pos_totals.values())
    print("\nWHERE do bad windows concentrate (across all failed trajectories' bad windows)?")
    print("| position | count | fraction |")
    print("|---|---|---|")
    for pos in ("leading", "interior", "trailing"):
        c = pos_totals.get(pos, 0)
        print(f"| {pos} | {c} | {c / total_bad:.1%} |" if total_bad else f"| {pos} | 0 | n/a |")

    recoverable = [d for d in failures if d["recoverable_from_correct_only_windows"]]
    print(f"\nof {n_fail} failures, recoverable-by-majority-vote-from-CORRECT-windows-only: "
         f"{len(recoverable)}/{n_fail} = {len(recoverable) / n_fail:.1%}")
    print("(high fraction here => reduction/guard logic is the bottleneck, not the classifier)")

    recoverable_1_2_bad = [d for d in recoverable if d["n_bad_windows"] in (1, 2)]
    print(f"of those recoverable cases, {len(recoverable_1_2_bad)} also had only 1-2 bad windows "
         f"({len(recoverable_1_2_bad) / n_fail:.1%} of ALL failures)")

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "n": args.n, "seed": args.seed, "n_eligible": n_total, "n_failed": n_fail,
        "bad_window_histogram": dict(bad_window_hist),
        "fraction_1_2_bad_windows": n_1_2_bad / n_fail if n_fail else None,
        "bad_window_position_totals": dict(pos_totals),
        "n_recoverable_from_correct_only": len(recoverable),
        "fraction_recoverable_from_correct_only": len(recoverable) / n_fail if n_fail else None,
        "decompositions": decompositions,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
