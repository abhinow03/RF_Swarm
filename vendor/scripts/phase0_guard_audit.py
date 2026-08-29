"""
V5 program, Phase 0, post-strategy-5-revert step 1 (docs/HISTORY.md): audit
stgt_bridge.py's _is_ambiguous_dispersed_converging guard directly against real
predictions, to check the user's hypothesis that it fires on windows where
dispersed/converging aren't even competitive.

The guard (stgt_bridge.py:114-120):

    def _is_ambiguous_dispersed_converging(class_probabilities: dict) -> bool:
        d, c = class_probabilities.get("dispersed"), class_probabilities.get("converging")
        return abs(d - c) < DISPERSED_CONVERGING_AMBIGUITY_MARGIN  # margin=0.15

This checks ONLY whether the two raw probabilities happen to be close to each other in
absolute terms -- not whether either of them is actually competitive for the window's
top prediction. With 8 classes, when the true/predicted class is something else entirely,
dispersed and converging are both typically small residual probabilities, which land close
to each other by chance far more often than the guard's docstring implies.

For every window across the same 509 pair-eligible trajectories used throughout Phase 0
(seed=999, identical to phase0_ceiling.py), records whether the guard fires, and whether
EITHER of dispersed/converging is in that window's top-2 predicted classes. Runs inference
only -- no training -- against whatever checkpoint is currently swarm_data/best_model.pt.

Usage:
    python scripts/phase0_guard_audit.py --n 1000
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.config import BASE_FORMATIONS, TRANSITION_CLASS  # noqa: E402
from swarm_intent.stgt_bridge import (  # noqa: E402
    _is_ambiguous_dispersed_converging, DISPERSED_CONVERGING_AMBIGUITY_MARGIN,
)
from swarm_intent.progress import Reporter  # noqa: E402
# 2026-08-09 step 26 (docs/V5_LOG.md): consolidated here from an inline, independently-
# maintained copy -- see src/swarm_intent/eval_trajectories.py for the full derivation.
from swarm_intent.eval_trajectories import (  # noqa: E402, F401
    sample_chain, build_long_sequence_labeled, ground_truth_pair,
)

DATA_DIR = REPO / "swarm_data"
CHECKPOINT = DATA_DIR / "best_model.pt"
SEED = 999


def build_long_sequence(chain: list[str], rng: np.random.Generator, spread: float, noise_std: float):
    """Thin wrapper: this script never needed per-timestep true labels, only the sequence."""
    return build_long_sequence_labeled(chain, rng, spread, noise_std)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    import torch
    from swarm_intent.stgt.config import device
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    print(f"checkpoint: epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f}")
    model = STGTModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    rng = np.random.default_rng(args.seed)
    reporter = Reporter("phase0_guard_audit", args.n, rate_hint=10.0)

    n_windows_total = 0
    n_guard_fires = 0
    n_guard_fires_neither_top2 = 0
    n_guard_fires_one_top2 = 0
    n_guard_fires_both_top2 = 0
    fire_examples = []  # a few examples of "neither in top-2" for the report

    for i in range(args.n):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        gt_pair = ground_truth_pair(chain)
        if gt_pair is None:
            reporter.update(1, item=f"seq {i} (skip)")
            continue

        predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)

        for w_idx, pred in enumerate(predictions):
            cp = pred.get("class_probabilities", {})
            if not cp:
                continue
            n_windows_total += 1
            fires = _is_ambiguous_dispersed_converging(cp)
            if not fires:
                continue
            n_guard_fires += 1
            ranked = sorted(cp.items(), key=lambda kv: kv[1], reverse=True)
            top2 = {ranked[0][0], ranked[1][0]} if len(ranked) >= 2 else {ranked[0][0]}
            d_in, c_in = "dispersed" in top2, "converging" in top2
            n_in_top2 = int(d_in) + int(c_in)
            if n_in_top2 == 0:
                n_guard_fires_neither_top2 += 1
                if len(fire_examples) < 8:
                    fire_examples.append({
                        "seq": i, "window": w_idx, "predicted_formation": pred["formation_type"],
                        "top2": ranked[:2], "dispersed_p": cp.get("dispersed"), "converging_p": cp.get("converging"),
                    })
            elif n_in_top2 == 1:
                n_guard_fires_one_top2 += 1
            else:
                n_guard_fires_both_top2 += 1

        reporter.update(1, item=f"seq {i}")

    reporter.status = "done"
    reporter._write()

    print("\n" + "=" * 100)
    print(f"dispersed_converging_ambiguity GUARD AUDIT (margin={DISPERSED_CONVERGING_AMBIGUITY_MARGIN}, "
         f"n_windows={n_windows_total})")
    print("=" * 100)
    print(f"guard fires on {n_guard_fires}/{n_windows_total} windows ({n_guard_fires/n_windows_total:.1%})")
    print(f"  of those firings:")
    print(f"    BOTH dispersed and converging in top-2 (genuinely competing): "
         f"{n_guard_fires_both_top2} ({n_guard_fires_both_top2/n_guard_fires:.1%})")
    print(f"    ONE of dispersed/converging in top-2: "
         f"{n_guard_fires_one_top2} ({n_guard_fires_one_top2/n_guard_fires:.1%})")
    print(f"    NEITHER dispersed nor converging in top-2 (spurious): "
         f"{n_guard_fires_neither_top2} ({n_guard_fires_neither_top2/n_guard_fires:.1%})")

    print("\nexamples of spurious firing (neither dispersed nor converging in top-2):")
    for ex in fire_examples:
        print(f"  seq={ex['seq']} window={ex['window']} predicted={ex['predicted_formation']} "
             f"top2={ex['top2']} dispersed_p={ex['dispersed_p']:.4f} converging_p={ex['converging_p']:.4f}")


if __name__ == "__main__":
    main()
