"""
Central configuration for the swarm-behaviour model.

This replaces the duplicated ``CFG`` / ``CFG_V2`` dicts that were copy-pasted
across the original notebooks. There is now ONE source of truth.

Usage
-----
    from swarm_intent.config import Config, formation_names

    cfg = Config()                 # 7-class default
    cfg_v2 = Config(n_classes=8)   # adds the "transitioning" class
    names = formation_names(cfg)   # ["v_shape", ..., ("transitioning")]
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List

# ---------------------------------------------------------------------------
# Canonical formation vocabulary. Index == integer label used by the model.
# The 8th class ("transitioning") is only active when Config.n_classes == 8.
# ---------------------------------------------------------------------------
BASE_FORMATIONS: List[str] = [
    "v_shape",
    "encirclement",
    "column",
    "diamond",
    "dispersed",
    "converging",
    "shield",
]
TRANSITION_CLASS = "transitioning"


@dataclass
class Config:
    """All tunable knobs in one place. Edit here, not in scattered cells."""

    # --- Graph construction ---
    # edge_threshold is in NORMALISED units (positions are normalised to ~[-3, 3]).
    edge_threshold: float = 2.0

    # --- GAT (spatial encoder) ---
    gat_in_dim: int = 3          # node features: (x, y, z)
    gat_hidden: int = 64
    gat_out_dim: int = 128
    gat_heads: int = 8
    gat_dropout: float = 0.1
    edge_dim: int = 4            # edge features: (dx, dy, dz, dist)

    # --- Temporal transformer ---
    d_model: int = 128           # must equal gat_out_dim
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 50

    # --- Output heads ---
    n_classes: int = 7           # set to 8 to enable the transitioning class
    n_reg: int = 3               # centroid_velocity, approach_rate, stability

    # --- Training ---
    batch_size: int = 64
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 12
    grad_clip: float = 1.0
    # Linear warmup for warmup_pct of total steps, then cosine decay down to
    # lr_min_frac * lr (a nonzero floor, not to ~0 like the previous OneCycleLR
    # default) -- see V5_LOG.md's strategy-6 entry: strategy 5's OneCycleLR run
    # (high peak lr, decay-to-near-zero) early-stopped at 22/150 with a visibly
    # volatile val curve, suggesting the schedule itself was destabilizing.
    warmup_pct: float = 0.1
    lr_min_frac: float = 0.05

    # --- Loss weights ---
    cls_weight: float = 1.0
    reg_weight: float = 0.1

    # --- Reproducibility / IO ---
    seed: int = 42
    data_dir: str = "swarm_data"

    def to_dict(self) -> dict:
        return asdict(self)


def formation_names(cfg: Config) -> List[str]:
    """Return the active label list for this config (7 or 8 classes)."""
    names = list(BASE_FORMATIONS)
    if cfg.n_classes == len(BASE_FORMATIONS) + 1:
        names.append(TRANSITION_CLASS)
    return names
