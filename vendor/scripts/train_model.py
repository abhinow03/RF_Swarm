"""Train the STGT swarm model on the saved splits.

Usage:
    python scripts/train_model.py --classes 8 --epochs 80
"""
import argparse

from swarm_intent.config import Config
from swarm_intent.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=7, choices=[7, 8])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--data-dir", default="swarm_data")
    ap.add_argument("--lr", type=float, default=None, help="override cfg.lr (peak LR)")
    ap.add_argument("--patience", type=int, default=None, help="override cfg.patience")
    ap.add_argument("--warmup-pct", type=float, default=None, help="override cfg.warmup_pct")
    ap.add_argument("--lr-min-frac", type=float, default=None, help="override cfg.lr_min_frac")
    args = ap.parse_args()

    kwargs = dict(n_classes=args.classes, epochs=args.epochs, data_dir=args.data_dir)
    if args.lr is not None:
        kwargs["lr"] = args.lr
    if args.patience is not None:
        kwargs["patience"] = args.patience
    if args.warmup_pct is not None:
        kwargs["warmup_pct"] = args.warmup_pct
    if args.lr_min_frac is not None:
        kwargs["lr_min_frac"] = args.lr_min_frac
    cfg = Config(**kwargs)
    train(cfg)


if __name__ == "__main__":
    main()
