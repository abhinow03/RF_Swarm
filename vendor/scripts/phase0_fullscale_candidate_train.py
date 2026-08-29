"""
Step 2 of the 2026-08-10 full-scale go/no-go (docs/V5_LOG.md step 51): trains the corrected-
port candidate at full scale -- n_per_formation=3000, n_transition=9000 hops (matches the
standing baseline's own 9000 independent transitioning draws 1:1, per step 50's validated,
un-rebalanced class ratio), robust=True normalization, acceleration uncapped, 150 epochs
(matching strategy 5/6's own full-scale convention). Single seed (100, fresh, disjoint from
every seed used elsewhere in this program) -- full-scale training is expensive, 5-seed
repetition was for the small-scale diagnostic, not required here per instruction (the full
1000-trajectory ceiling comparison in step 52 is the actual statistical gate).

Does NOT touch swarm_data/ (the standing strategy-5 checkpoint/dataset) -- writes to
swarm_data_candidate_fullscale/ instead.

Usage (run inside tmux -- one full-scale training run, potentially 30-90+ min):
    python scripts/phase0_fullscale_candidate_train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.config import Config  # noqa: E402
from swarm_intent.data import generate_dataset, save_splits, split_and_normalize  # noqa: E402
from swarm_intent.train import train, get_device  # noqa: E402

SEED = 100
DATA_DIR = REPO / "swarm_data_candidate_fullscale"
N_PER_FORMATION = 3000
N_TRANSITION = 9000
EPOCHS = 150


def main():
    print("=" * 100)
    print("PHASE A: generate full-scale candidate dataset")
    print("=" * 100)
    cfg = Config(seed=SEED)
    X, y, names, diag = generate_dataset(
        cfg, n_per_formation=N_PER_FORMATION, n_timesteps=50, include_transitions=True,
        n_transition=N_TRANSITION, corrected_blend_timing=True, windowed_examples=True,
        content_majority_labeling=True, return_diagnostics=True,
    )
    print(f"X.shape={X.shape}, hops={diag['n_hops_sampled']}, kept_transitioning={diag['n_examples_kept']}, "
         f"excluded={diag['n_excluded']}")

    DATA_DIR.mkdir(exist_ok=True)
    cfg2 = Config(seed=SEED, n_classes=8, data_dir=str(DATA_DIR))
    splits = split_and_normalize(X, y, cfg2, robust=True)
    save_splits(splits, cfg2)
    np.save(DATA_DIR / "train_mean.npy", np.array(splits["train_mean"], dtype=np.float32))
    np.save(DATA_DIR / "train_std.npy", np.array(splits["train_std"], dtype=np.float32))
    print(f"train_mean={splits['train_mean']:.3f} train_std={splits['train_std']:.3f} (robust=True)")

    print("\n" + "=" * 100)
    print(f"PHASE B: train candidate, {EPOCHS} epochs, seed={SEED}")
    print("=" * 100)
    device = get_device()
    torch.manual_seed(SEED)
    train_cfg = Config(seed=SEED, n_classes=8, data_dir=str(DATA_DIR), epochs=EPOCHS)
    model, history, test_acc = train(train_cfg, device=device, ckpt_name="best_model.pt")
    print(f"\nFINAL test_acc={test_acc:.4f}")
    print(f"checkpoint saved to {DATA_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
