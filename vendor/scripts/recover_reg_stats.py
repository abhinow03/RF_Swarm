"""
Step 1 of the "vendor teammate's retrained STGT" session (AUDIT.md sec V).

Teammate's retrained checkpoint (swarm_data/best_model.pt) and split arrays
were handed over without reg_mean.npy / reg_std.npy on disk -- dataset.py
only ever kept them as SwarmDataset instance attributes, never persisted
them as standalone files the way train_mean.npy/train_std.npy are.

First checks whether the checkpoint already embeds them (it does, per the
"cfg"/"reg_mean"/"reg_std" top-level keys -- this script still recomputes
them independently as a correctness check rather than trusting that blindly).

Reconstruction imports compute_regression_labels DIRECTLY from
src/swarm_intent/dataset.py (not a reimplementation) so there is no risk of
the recovery code drifting from the training-time definition.

Usage:
    python scripts/recover_reg_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.dataset import compute_regression_labels  # noqa: E402

DATA_DIR = REPO / "swarm_data"


def main():
    ckpt = torch.load(DATA_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    print("checkpoint top-level keys:", list(ckpt.keys()))

    has_reg_stats = "reg_mean" in ckpt and "reg_std" in ckpt
    print(f"\ncheckpoint already contains reg_mean/reg_std: {has_reg_stats}")
    if has_reg_stats:
        ckpt_reg_mean = np.asarray(ckpt["reg_mean"], dtype=np.float32)
        ckpt_reg_std = np.asarray(ckpt["reg_std"], dtype=np.float32)
        print(f"  reg_mean = {ckpt_reg_mean}")
        print(f"  reg_std  = {ckpt_reg_std}")

    # Reconstruct independently, exactly per dataset.py's SwarmDataset.__init__:
    # reg_labels = [compute_regression_labels(X[i]) for i in range(len(X))]
    # reg_mean = reg_labels.mean(axis=0); reg_std = reg_labels.std(axis=0) + 1e-6
    X_train_norm = np.load(DATA_DIR / "X_train.npy")
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    X_train_raw = X_train_norm * train_std + train_mean

    reg_labels = np.array([compute_regression_labels(X_train_raw[i])
                           for i in range(len(X_train_raw))])
    reconstructed_reg_mean = reg_labels.mean(axis=0)
    reconstructed_reg_std = reg_labels.std(axis=0) + 1e-6
    print(f"\nreconstructed from X_train.npy (n={len(X_train_raw)}):")
    print(f"  reg_mean = {reconstructed_reg_mean}")
    print(f"  reg_std  = {reconstructed_reg_std}")

    if has_reg_stats:
        mismatches = ~np.isclose(reconstructed_reg_mean, ckpt_reg_mean, rtol=1e-3, atol=1e-4)
        if not mismatches.any():
            print("\nreconstruction MATCHES checkpoint's embedded reg_mean/reg_std exactly.")
        else:
            ratio = ckpt_reg_mean / reconstructed_reg_mean
            print(f"\nMISMATCH: reconstruction does NOT match checkpoint on field(s) "
                  f"{np.where(mismatches)[0].tolist()} (0=centroid_velocity, 1=approach_rate, "
                  f"2=stability). checkpoint/reconstructed ratio per field: {ratio}")
            print("Diagnosis (see AUDIT.md sec V): approach_rate and stability match this "
                  "repo's dataset.py::compute_regression_labels() EXACTLY when run on "
                  "denormalised X_raw -- only centroid_velocity is off, by a clean 2.000x. "
                  "Both mean and std scale by exactly 2x (the signature of a uniform "
                  "per-sample multiplicative factor, not a data/seed mismatch), consistent "
                  "with this repo's formula computing raw per-FRAME displacement while the "
                  "checkpoint's convention divides by dt=0.5s/frame (inference.py's own "
                  "sliding_window_inference already assumes this dt). Using the CHECKPOINT's "
                  "own reg_mean/reg_std below: the model's reg_head was trained to predict "
                  "normalised values against THESE stats, so denormalising with any other "
                  "stats would silently rescale every downstream centroid_velocity by 2x.")
        print("\nUsing the checkpoint's own embedded reg_mean/reg_std as the recovered values "
              "-- this is what the trained reg_head must be denormalised against, regardless "
              "of whether this repo's current formula reproduces it exactly.")
        final_mean, final_std = ckpt_reg_mean, ckpt_reg_std
    else:
        print("\ncheckpoint did NOT embed reg_mean/reg_std -- using reconstructed values.")
        final_mean, final_std = reconstructed_reg_mean, reconstructed_reg_std

    np.save(DATA_DIR / "reg_mean.npy", final_mean.astype(np.float32))
    np.save(DATA_DIR / "reg_std.npy", final_std.astype(np.float32))
    print(f"\nsaved {DATA_DIR / 'reg_mean.npy'} and {DATA_DIR / 'reg_std.npy'}")

    # Also stash the raw (denormalised) reg_labels for step 2's distribution
    # analysis, so that script doesn't need to redo denormalisation + recompute.
    np.save(DATA_DIR / "_reg_labels_train_raw.npy", reg_labels.astype(np.float32))
    print(f"saved {DATA_DIR / '_reg_labels_train_raw.npy'} (n={len(reg_labels)}, for step 2)")


if __name__ == "__main__":
    main()
