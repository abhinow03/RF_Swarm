"""Generate the synthetic swarm dataset and save train/val/test splits.

Usage:
    python scripts/generate_data.py --per-formation 1000 --transitions 2000
"""
import argparse

from swarm_intent.config import Config
from swarm_intent.data import generate_dataset, split_and_normalize, save_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-formation", type=int, default=1000)
    ap.add_argument("--transitions", type=int, default=0,
                    help="number of transition sequences (adds the 8th class)")
    ap.add_argument("--data-dir", default="swarm_data")
    args = ap.parse_args()

    include = args.transitions > 0
    cfg = Config(n_classes=8 if include else 7, data_dir=args.data_dir)
    X, y, names = generate_dataset(
        cfg, n_per_formation=args.per_formation,
        include_transitions=include, n_transition=args.transitions,
    )
    print(f"Generated {len(X)} sequences across {len(names)} classes: {names}")
    splits = split_and_normalize(X, y, cfg)
    save_splits(splits, cfg)
    print(f"Saved splits to {cfg.data_dir}/ "
          f"(train_mean={splits['train_mean']:.3f}, train_std={splits['train_std']:.3f})")


if __name__ == "__main__":
    main()
