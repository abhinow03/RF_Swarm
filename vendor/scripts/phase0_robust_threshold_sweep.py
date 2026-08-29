"""
V5 program, Phase 0, post-guard-audit step 5 (docs/HISTORY.md): `robust=True`'s
majority-vote threshold (`DEFAULT_ROBUST_THRESHOLD=0.7`, `stgt_bridge.py`) was
tuned once, on the dev split, BEFORE this session's guard fix and full guard
audit. Both changed the underlying signal the threshold operates on (the guard
fix changed which windows get flagged unknown; step 4's audit found the trim
step itself discards genuine signal 62.5% of the time). Re-sweeps the
threshold against the CURRENT pipeline, coverage vs precision at each point,
tuned on the dev split ONLY (seed=1, explicitly disjoint from the seed=999
held-out population every other Phase 0 measurement uses) -- then reports the
chosen threshold's performance on the held-out seed=999 population too, as a
standing check against overfitting the threshold to the dev split.

Usage (run inside tmux):
    python scripts/phase0_robust_threshold_sweep.py --n-dev 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from swarm_intent.coverage import classify_observation, BUCKET_A  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from phase0_decompose_failures import sample_chain, build_long_sequence_labeled, ground_truth_pair  # noqa: E402

DATA_DIR = REPO / "swarm_data"
CHECKPOINT = DATA_DIR / "best_model.pt"
DEV_SEED = 1  # disjoint from seed=999 (held-out) and seed=0 (an earlier engagement's own eval seed)
HELD_OUT_SEED = 999
THRESHOLD_SWEEP = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]


def generate_predictions(model, ckpt, reg_mean, reg_std, train_mean, train_std, n, seed):
    rng = np.random.default_rng(seed)
    reporter = Reporter(f"phase0_robust_sweep_gen_seed{seed}", n, rate_hint=8.0)
    from swarm_intent.stgt.inference import sliding_window_inference
    records = []
    for i in range(n):
        chain = [str(f) for f in sample_chain(rng)]
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq, _ = build_long_sequence_labeled(chain, rng, spread, noise_std)
        gt_pair = ground_truth_pair(chain)
        if gt_pair is not None:
            predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                                   train_mean, train_std, window_size=50, stride=10, dt=0.5)
            records.append({"gt_pair": list(gt_pair), "predictions": predictions})
        reporter.update(1, item=f"seq {i}")
    reporter.status = "done"
    reporter._write()
    return records


def sweep(records, thresholds):
    results = []
    for t in thresholds:
        n_recovered = n_correct = 0
        for r in records:
            info = classify_observation(r["predictions"], robust=True, robust_threshold=t)
            if info["bucket"] == BUCKET_A:
                n_recovered += 1
                if tuple(info["rules_key"]) == tuple(r["gt_pair"]):
                    n_correct += 1
        n = len(records)
        results.append({"threshold": t, "n": n, "n_recovered": n_recovered,
                        "coverage": n_recovered / n, "n_correct": n_correct,
                        "precision": n_correct / n_recovered if n_recovered else 0.0})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dev", type=int, default=500)
    ap.add_argument("--n-held-out", type=int, default=1000)
    args = ap.parse_args()

    import torch
    from swarm_intent.stgt.config import device
    from swarm_intent.stgt.model import STGTModel

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    print(f"checkpoint: epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f}")
    model = STGTModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    print(f"\n=== generating DEV split (seed={DEV_SEED}, n={args.n_dev}) -- tuning happens HERE ONLY ===")
    dev_records = generate_predictions(model, ckpt, reg_mean, reg_std, train_mean, train_std,
                                       args.n_dev, DEV_SEED)
    print(f"dev pair-eligible n={len(dev_records)}")

    dev_results = sweep(dev_records, THRESHOLD_SWEEP)
    print("\n" + "=" * 100)
    print("ROBUST-REDUCTION THRESHOLD SWEEP -- DEV SPLIT ONLY (tuning happens here)")
    print("=" * 100)
    print("| threshold | n_recovered | coverage | n_correct | precision |")
    print("|---|---|---|---|---|")
    for r in dev_results:
        print(f"| {r['threshold']} | {r['n_recovered']}/{r['n']} | {r['coverage']:.1%} | "
             f"{r['n_correct']}/{r['n_recovered'] if r['n_recovered'] else 1} | {r['precision']:.1%} |")

    # Selection rule, stated up front: the LOWEST threshold whose precision is
    # within 3 points of the highest precision measured anywhere in the sweep
    # (a plateau-aware rule -- avoids picking an unnecessarily strict threshold
    # once precision stops improving, matching sec AG's original convention but
    # relaxed from a hard 90% floor, which nothing in this sweep is expected to
    # reach given step 4's finding that the trim step itself has a 62.5%
    # spurious rate independent of the vote threshold).
    max_precision = max(r["precision"] for r in dev_results if r["n_recovered"] > 0)
    eligible = [r for r in dev_results if r["n_recovered"] > 0 and r["precision"] >= max_precision - 0.03]
    chosen = min(eligible, key=lambda r: r["threshold"])
    print(f"\n=== CHOSEN THRESHOLD: {chosen['threshold']} ===")
    print(f"dev coverage={chosen['coverage']:.1%}, dev precision={chosen['precision']:.1%} "
         f"(selection rule: lowest threshold within 3pt of the sweep's max precision "
         f"{max_precision:.1%})")

    print(f"\n=== generating HELD-OUT split (seed={HELD_OUT_SEED}, n={args.n_held_out}) -- "
         f"confirmation only, chosen threshold is FROZEN ===")
    held_out_records = generate_predictions(model, ckpt, reg_mean, reg_std, train_mean, train_std,
                                            args.n_held_out, HELD_OUT_SEED)
    print(f"held-out pair-eligible n={len(held_out_records)}")
    held_out_at_chosen = sweep(held_out_records, [chosen["threshold"]])[0]
    held_out_at_default = sweep(held_out_records, [0.7])[0]

    print(f"\nheld-out @ chosen threshold ({chosen['threshold']}): "
         f"coverage={held_out_at_chosen['coverage']:.1%}, precision={held_out_at_chosen['precision']:.1%}")
    print(f"held-out @ current shipped default (0.7): "
         f"coverage={held_out_at_default['coverage']:.1%}, precision={held_out_at_default['precision']:.1%}")
    print(f"dev-vs-held-out precision gap at chosen threshold: "
         f"{abs(chosen['precision'] - held_out_at_chosen['precision']):.1%}")

    out = {"dev_seed": DEV_SEED, "held_out_seed": HELD_OUT_SEED, "dev_sweep": dev_results,
          "chosen_threshold": chosen["threshold"], "chosen_dev": chosen,
          "held_out_at_chosen": held_out_at_chosen, "held_out_at_current_default_0.7": held_out_at_default}
    (REPO / "evaluation" / "phase0_robust_threshold_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved evaluation/phase0_robust_threshold_sweep.json")


if __name__ == "__main__":
    main()
