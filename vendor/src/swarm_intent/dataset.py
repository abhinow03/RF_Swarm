"""
PyTorch dataset, regression pseudo-labels, and collation.

Fix versus the original: ``SwarmDataset`` can now receive ``reg_mean`` /
``reg_std`` from the training set so val/test use the SAME normalisation
(the original gave each split its own stats).

CAVEAT (documented design issue, kept for compatibility): the regression
pseudo-labels are computed analytically from the same positions the model
sees as input, so they are a closed-form function of the input rather than
independent ground truth. See CODE_REVIEW.md. Consider deriving them from
simulation metadata (true speed/spread) instead.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config
from .graph import sequence_to_graphs


def compute_regression_labels(seq: np.ndarray) -> np.ndarray:
    """[centroid_velocity, approach_rate, formation_stability] from a (T,6,3) seq.

    NOTE: computed on NORMALISED positions, so units are normalised-space, not
    physical m/s. Do not present these to the operator as m/s without rescaling.
    """
    centroids = seq.mean(axis=1)
    deltas = np.diff(centroids, axis=0)
    centroid_velocity = np.linalg.norm(deltas, axis=1).mean()

    centered = seq - centroids[:, None, :]
    spread = np.linalg.norm(centered, axis=2).mean(axis=1)
    t = np.arange(seq.shape[0], dtype=float)
    approach_rate = np.polyfit(t, spread, deg=1)[0]

    stability = 1.0 - (spread.std() / (spread.mean() + 1e-6))
    stability = float(np.clip(stability, 0.0, 1.0))
    return np.array([float(centroid_velocity), float(approach_rate), stability],
                    dtype=np.float32)


class SwarmDataset(Dataset):
    """Wraps numpy arrays and pre-builds the per-timestep graphs."""

    def __init__(self, X, y, cfg: Config, precompute: bool = True,
                 reg_mean=None, reg_std=None):
        self.y = torch.tensor(y, dtype=torch.long)
        self.cfg = cfg

        reg_labels = np.array([compute_regression_labels(X[i]) for i in range(len(X))])
        # Use provided (train) stats if given, else compute from this split.
        self.reg_mean = reg_labels.mean(axis=0) if reg_mean is None else reg_mean
        self.reg_std = (reg_labels.std(axis=0) + 1e-6) if reg_std is None else reg_std
        reg_norm = (reg_labels - self.reg_mean) / self.reg_std
        self.reg_labels = torch.tensor(reg_norm, dtype=torch.float32)

        if precompute:
            self.graphs = [
                sequence_to_graphs(X[i], cfg.edge_threshold) for i in range(len(X))
            ]
            self.X = None
        else:
            self.X = X
            self.graphs = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.graphs is not None:
            return self.graphs[idx], self.y[idx], self.reg_labels[idx]
        return (sequence_to_graphs(self.X[idx], self.cfg.edge_threshold),
                self.y[idx], self.reg_labels[idx])


def collate_fn(batch):
    """Keep graph-lists as lists; stack labels and regression targets."""
    graph_seqs, labels, reg_labels = zip(*batch)
    return list(graph_seqs), torch.stack(labels), torch.stack(reg_labels)
