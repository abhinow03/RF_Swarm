"""
V5 program, Phase 0, step 1 of the discipline-catch turn (docs/V5_LOG.md 2026-08-09
step 0): seed=1 (the previous "dev split") was used for threshold-tuning SELECTION
twice across two separate sessions -- a real discipline lapse even though neither
individual round's selection step read the seed=999 ceiling battery. This cuts a
FRESH, single-use mining split and re-runs the threshold sweep there. Standing rule
recorded this session: a mining split is used for tuning ONCE, then retired.

MINING_SEED=2024, disjoint from every other seed with a role in this project:
  - 42: cfg.seed, the STGT train/val/test split itself
  - 999: the standing 1000-trajectory ceiling battery (phase0_ceiling.py) -- the
    number HALT GATE 1 is judged against, NEVER used for tuning
  - 1: the RETIRED dev split (used twice already, sec AG + prior turn's step 5) --
    not reused here, and should not be reused again after this turn either

This split is used for exactly one thing (the sweep below) and should be considered
spent afterward, the same as seed=1 now is.

Usage (run inside tmux):
    python scripts/phase0_mining_split_sweep.py --n-mining 500
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
MINING_SEED = 2024  # fresh, single-use -- see module docstring
HELD_OUT_SEED = 999
RETIRED_SEEDS = (1,)  # not used here, recorded so a future reader knows not to reuse them either
THRESHOLD_SWEEP = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]


def generate_predictions(model, ckpt, reg_mean, reg_std, train_mean, train_std, n, seed, label):
    rng = np.random.default_rng(seed)
    reporter = Reporter(f"phase0_mining_sweep_gen_{label}", n, rate_hint=8.0)
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
    ap.add_argument("--n-mining", type=int, default=500)
    ap.add_argument("--n-held-out", type=int, default=1000)
    args = ap.parse_args()

    print(f"MINING_SEED={MINING_SEED} (fresh, single-use), HELD_OUT_SEED={HELD_OUT_SEED} "
         f"(confirmation only), RETIRED_SEEDS={RETIRED_SEEDS} (not touched)")

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

    print(f"\n=== generating MINING split (seed={MINING_SEED}, n={args.n_mining}) -- "
         f"tuning happens HERE ONLY, then this split is retired ===")
    mining_records = generate_predictions(model, ckpt, reg_mean, reg_std, train_mean, train_std,
                                          args.n_mining, MINING_SEED, "mining")
    print(f"mining pair-eligible n={len(mining_records)}")

    mining_results = sweep(mining_records, THRESHOLD_SWEEP)
    print("\n" + "=" * 100)
    print("ROBUST-REDUCTION THRESHOLD SWEEP -- FRESH MINING SPLIT (seed=2024) ONLY")
    print("=" * 100)
    print("| threshold | n_recovered | coverage | n_correct | precision |")
    print("|---|---|---|---|---|")
    for r in mining_results:
        print(f"| {r['threshold']} | {r['n_recovered']}/{r['n']} | {r['coverage']:.1%} | "
             f"{r['n_correct']}/{r['n_recovered'] if r['n_recovered'] else 1} | {r['precision']:.1%} |")

    max_precision = max(r["precision"] for r in mining_results if r["n_recovered"] > 0)
    eligible = [r for r in mining_results if r["n_recovered"] > 0 and r["precision"] >= max_precision - 0.03]
    chosen = min(eligible, key=lambda r: r["threshold"])
    print(f"\n=== CHOSEN THRESHOLD (mining split): {chosen['threshold']} ===")
    print(f"mining coverage={chosen['coverage']:.1%}, mining precision={chosen['precision']:.1%} "
         f"(selection rule: lowest threshold within 3pt of the sweep's max precision "
         f"{max_precision:.1%} -- same rule as the prior, now-retired sweep)")

    current_default_on_mining = [r for r in mining_results if r["threshold"] == 0.7][0]
    print(f"\ncurrent shipped default (0.7) on mining split: "
         f"coverage={current_default_on_mining['coverage']:.1%}, "
         f"precision={current_default_on_mining['precision']:.1%}")
    dominates = (chosen["coverage"] >= current_default_on_mining["coverage"]
                and chosen["precision"] >= current_default_on_mining["precision"]
                and chosen["threshold"] != 0.7)
    print(f"does chosen threshold Pareto-dominate the current default on THIS split? {dominates}")

    print(f"\n=== generating HELD-OUT split (seed={HELD_OUT_SEED}, n={args.n_held_out}) -- "
         f"confirmation only, chosen threshold is FROZEN before this generation ===")
    held_out_records = generate_predictions(model, ckpt, reg_mean, reg_std, train_mean, train_std,
                                            args.n_held_out, HELD_OUT_SEED, "heldout")
    print(f"held-out pair-eligible n={len(held_out_records)}")
    held_out_at_chosen = sweep(held_out_records, [chosen["threshold"]])[0]
    held_out_at_default = sweep(held_out_records, [0.7])[0]

    print(f"\nheld-out @ chosen threshold ({chosen['threshold']}): "
         f"coverage={held_out_at_chosen['coverage']:.1%}, precision={held_out_at_chosen['precision']:.1%}")
    print(f"held-out @ current shipped default (0.7): "
         f"coverage={held_out_at_default['coverage']:.1%}, precision={held_out_at_default['precision']:.1%}")
    print(f"mining-vs-held-out precision gap at chosen threshold: "
         f"{abs(chosen['precision'] - held_out_at_chosen['precision']):.1%}")

    out = {"mining_seed": MINING_SEED, "held_out_seed": HELD_OUT_SEED, "retired_seeds": list(RETIRED_SEEDS),
          "mining_sweep": mining_results, "chosen_threshold": chosen["threshold"], "chosen_mining": chosen,
          "current_default_on_mining": current_default_on_mining, "dominates_current_default": dominates,
          "held_out_at_chosen": held_out_at_chosen, "held_out_at_current_default_0.7": held_out_at_default}
    (REPO / "evaluation" / "phase0_mining_split_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved evaluation/phase0_mining_split_sweep.json")
    print(f"\nMINING_SEED={MINING_SEED} is now RETIRED -- do not reuse for future tuning decisions.")


if __name__ == "__main__":
    main()
