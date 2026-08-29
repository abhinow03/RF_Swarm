"""Thin wrapper around the peer STGT formation classifier (torch + torch_geometric).

Loads the trained model ONCE and classifies a (T, 6, 3) pose-sequence window into
one of the 8 STGT formation classes. Everything torch is imported lazily here so the
rest of the package (and colcon/flake) does not require torch to be present.

Canonical artifacts (validated 2026-08-26, see phase5-status memory):
  best_model.pt + train_mean/std (scalar 32.408 / 68.177). Classification uses only
  the model + train stats; reg_mean/std affect ONLY the regression head (velocity /
  approach-rate / stability), which this classifier does not publish — so dummy reg
  stats are fine here (the STGT-regression PATH prediction handles reg stats itself).
"""
from __future__ import annotations

import os
import numpy as np


class FormationClassifier:
    def __init__(self, model_dir: str, checkpoint: str = 'best_model.pt', device: str = 'cpu'):
        import torch                                    # lazy
        from swarm_intent.stgt.config import CFG, FORMATION_NAMES
        from swarm_intent.stgt.model import STGTModel

        self._torch = torch
        self.CFG = CFG
        self.FORMATION_NAMES = list(FORMATION_NAMES)
        self.window = int(CFG['max_seq_len'])           # 50
        self.n_nodes = 6                                # model trained on 6 drones

        self._tm = float(np.load(os.path.join(model_dir, 'train_mean.npy')).reshape(-1)[0])
        self._ts = float(np.load(os.path.join(model_dir, 'train_std.npy')).reshape(-1)[0])

        self._dev = torch.device(device)
        self.model = STGTModel(CFG).to(self._dev)
        ckpt = torch.load(os.path.join(model_dir, checkpoint),
                          map_location=self._dev, weights_only=False)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        self.model.load_state_dict(sd)
        self.model.eval()
        # reg stats: the checkpoint EMBEDS its own matched reg_mean/reg_std (the
        # external swarm_data files are STALE) — prefer embedded, else neutral.
        # Classification does not use these; only the regression head does.
        if isinstance(ckpt, dict) and 'reg_mean' in ckpt and 'reg_std' in ckpt:
            self._rm = np.asarray(ckpt['reg_mean']).ravel()
            self._rs = np.asarray(ckpt['reg_std']).ravel()
        else:
            self._rm, self._rs = np.zeros(3), np.ones(3)

    def classify(self, seq: np.ndarray) -> dict:
        """seq: (T, 6, 3) ENU metres. Returns predict_v2's dict (formation_type,
        formation_confidence, class_probabilities, ...). Normalizes with train stats."""
        from swarm_intent.stgt.inference import predict_v2
        seq = np.asarray(seq, dtype=float)
        seq_norm = (seq - self._tm) / self._ts
        return predict_v2(self.model, seq_norm, self.CFG, self._rm, self._rs,
                          self.FORMATION_NAMES)

    def prob_vector(self, out: dict) -> list:
        """class_probabilities dict -> list ordered by FORMATION_NAMES (length 8)."""
        cp = out['class_probabilities']
        return [float(cp.get(name, 0.0)) for name in self.FORMATION_NAMES]
