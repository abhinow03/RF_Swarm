"""
V5 program, Phase 0 steps 2-4 (docs/V5_LOG.md / docs/V5_STATE.json): measure the ceiling
end-to-end tactical accuracy is upper-bounded by -- if the retrained STGT recovers the wrong
(a, b) pair, RULES is keyed wrong and no downstream LLM layer can fix it.

Reuses llm_finetuning/measure_coverage.py's exact sampling regime (sample_chain,
per-hop segment length ~ U{50,100}, blend_start/end fractions, spread/noise_std ranges) for
methodological consistency with every other coverage measurement in this project, but adds
per-timestep TRUE label tracking (build_long_sequence's original design dropped this when
migrated from upstream's generate_transition_sequence, which returned per_step_labels) so
per-window formation accuracy and a full confusion matrix can be scored, not just the
pair-level bucket classification measure_coverage.py already does.

Seed=999 -- explicitly disjoint from every seed used anywhere earlier in this project's
history (seed=0: sec AE/AF/AG/AH's 500-sequence real-output evals; seed=1: sec AG's
threshold-tuning dev split; seed=42: cfg.seed, the STGT train/val/test split itself).

Three things reported, per the plan's step 3:
  1. Per-class formation accuracy (window-level: predicted class vs. majority-vote true
     label over each window's timestep span).
  2. Full 8x8 confusion matrix (7 BASE_FORMATIONS + "transitioning").
  3. PAIR-LEVEL accuracy: for the subset of trajectories whose true chain is length 1
     (steady state) or 2 (single resolvable transition) -- the only chains RULES can ever
     answer -- does stgt_bridge's (robust=False, the shipped default) reduced (a, b) pair
     match the generator's own known chain. This is the ceiling.

Usage (run inside tmux):
    python scripts/phase0_ceiling.py --n 1000
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
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.config import BASE_FORMATIONS, TRANSITION_CLASS  # noqa: E402
from swarm_intent.coverage import classify_observation, BUCKET_A  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402
# 2026-08-09 step 26 (docs/V5_LOG.md): consolidated here from an inline, independently-
# maintained copy that had already silently diverged from
# scripts/phase0_decompose_failures.py's (separately, since-fixed) version -- see
# src/swarm_intent/eval_trajectories.py for the full derivation. This means
# phase0_ceiling.py's own HALT GATE 1 pooled-ceiling number will change if re-run against
# this now-fixed formula; that is the intended, disclosed consequence of consolidating
# rather than leaving 4 copies free to diverge silently again.
from swarm_intent.eval_trajectories import (  # noqa: E402, F401
    sample_chain, build_long_sequence_labeled, ground_truth_pair,
)

DATA_DIR = REPO / "swarm_data"
CHECKPOINT = DATA_DIR / "best_model.pt"
SEED = 999
CLASS_ORDER = list(BASE_FORMATIONS) + [TRANSITION_CLASS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(REPO / "evaluation" / "phase0_ceiling.json"))
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
    reporter = Reporter("phase0_ceiling", args.n, rate_hint=2.0)

    confusion = Counter()  # (true, pred) -> count
    per_class_total = Counter()
    pair_records = []
    n_windows_total = 0

    for i in range(args.n):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq, true_labels = build_long_sequence_labeled(chain, rng, spread, noise_std)

        predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)

        window_size, stride = 50, 10
        for w_idx, pred in enumerate(predictions):
            start = w_idx * stride
            end = min(start + window_size, len(true_labels))
            window_true_labels = true_labels[start:end]
            if not window_true_labels:
                continue
            true_label = Counter(window_true_labels).most_common(1)[0][0]
            pred_label = pred["formation_type"] if pred["formation_type"] in CLASS_ORDER else TRANSITION_CLASS
            confusion[(true_label, pred_label)] += 1
            per_class_total[true_label] += 1
            n_windows_total += 1

        gt_pair = ground_truth_pair(chain)
        if gt_pair is not None:
            bucket_info = classify_observation(predictions, robust=False)
            recovered = tuple(bucket_info["rules_key"]) if bucket_info["bucket"] == BUCKET_A else None
            bucket_info_r = classify_observation(predictions, robust=True)
            recovered_r = tuple(bucket_info_r["rules_key"]) if bucket_info_r["bucket"] == BUCKET_A else None
            pair_records.append({
                "i": i, "true_chain": chain, "gt_pair": list(gt_pair),
                "bucket": bucket_info["bucket"],
                "recovered_pair": list(recovered) if recovered else None,
                "pair_correct": (recovered == gt_pair) if recovered else False,
                "bucket_robust": bucket_info_r["bucket"],
                "recovered_pair_robust": list(recovered_r) if recovered_r else None,
                "pair_correct_robust": (recovered_r == gt_pair) if recovered_r else False,
            })

        reporter.update(1, item=f"seq {i}")

    reporter.status = "done"
    reporter._write()

    # ---- per-class accuracy + confusion matrix ----
    print("\n" + "=" * 100)
    print(f"PER-CLASS FORMATION ACCURACY (window-level, n_windows={n_windows_total})")
    print("=" * 100)
    print("| true class | n | accuracy |")
    print("|---|---|---|")
    per_class_acc = {}
    for c in CLASS_ORDER:
        total = per_class_total.get(c, 0)
        correct = confusion.get((c, c), 0)
        acc = correct / total if total else float("nan")
        per_class_acc[c] = acc
        print(f"| {c} | {total} | {acc:.1%} |" if total else f"| {c} | 0 | n/a |")
    overall_window_acc = sum(confusion.get((c, c), 0) for c in CLASS_ORDER) / n_windows_total

    print(f"\noverall window-level accuracy: {overall_window_acc:.1%} (n={n_windows_total})")

    print("\n" + "=" * 100)
    print("FULL 8x8 CONFUSION MATRIX (rows=true, cols=predicted)")
    print("=" * 100)
    header = "| true\\pred | " + " | ".join(CLASS_ORDER) + " |"
    print(header)
    print("|" + "---|" * (len(CLASS_ORDER) + 1))
    for t in CLASS_ORDER:
        row = [str(confusion.get((t, p), 0)) for p in CLASS_ORDER]
        print(f"| {t} | " + " | ".join(row) + " |")

    # ---- pair-level accuracy (the ceiling) ----
    n_pair_eligible = len(pair_records)
    n_pair_correct = sum(1 for r in pair_records if r["pair_correct"])
    pair_accuracy = n_pair_correct / n_pair_eligible if n_pair_eligible else float("nan")
    bucket_dist = Counter(r["bucket"] for r in pair_records)

    print("\n" + "=" * 100)
    print(f"PAIR-LEVEL ACCURACY (the ceiling) -- n_eligible={n_pair_eligible} "
         f"(chains of length 1 or 2 out of {args.n} total)")
    print("=" * 100)
    print(f"pair-level accuracy (robust=False, shipped default): {n_pair_correct}/{n_pair_eligible} = {pair_accuracy:.1%}")
    print(f"bucket distribution (robust=False): {dict(bucket_dist)}")

    n_pair_correct_r = sum(1 for r in pair_records if r["pair_correct_robust"])
    pair_accuracy_r = n_pair_correct_r / n_pair_eligible if n_pair_eligible else float("nan")
    bucket_dist_r = Counter(r["bucket_robust"] for r in pair_records)
    print(f"pair-level accuracy (robust=True, sec AG's majority-vote reduction): "
         f"{n_pair_correct_r}/{n_pair_eligible} = {pair_accuracy_r:.1%}")
    print(f"bucket distribution (robust=True): {dict(bucket_dist_r)}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "n": args.n, "seed": args.seed,
        "per_class_accuracy": per_class_acc,
        "overall_window_accuracy": overall_window_acc,
        "confusion_matrix": {f"{t}|{p}": confusion.get((t, p), 0) for t in CLASS_ORDER for p in CLASS_ORDER},
        "n_pair_eligible": n_pair_eligible, "n_pair_correct": n_pair_correct,
        "pair_level_accuracy": pair_accuracy,
        "pair_bucket_distribution": dict(bucket_dist),
        "n_pair_correct_robust": n_pair_correct_r,
        "pair_level_accuracy_robust": pair_accuracy_r,
        "pair_bucket_distribution_robust": dict(bucket_dist_r),
        "pair_records": pair_records,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
