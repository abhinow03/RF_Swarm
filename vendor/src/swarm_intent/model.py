"""
Spatial-Temporal Graph Transformer (STGT) for swarm formation classification.

  SpatialGAT          : (6,3) positions -> (d_model,) per-timestep graph vector
  PositionalEncoding  : sinusoidal time encoding
  TemporalTransformer : (T, d_model) -> (d_model,) sequence vector
  STGTModel           : full model with classification + regression heads

Migrated from the notebooks. Change from the original: the model no longer
relies on a module-level global ``device`` — it infers its device from its own
parameters, so the same code runs on CPU or GPU without edits.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch

from .config import Config


class SpatialGAT(nn.Module):
    """Two-layer GATv2 mapping (6, 3) positions to a (gat_out_dim,) embedding."""

    def __init__(self, cfg: Config):
        super().__init__()
        heads_1 = cfg.gat_heads
        hidden_per_head = cfg.gat_hidden // heads_1

        self.conv1 = GATv2Conv(
            in_channels=cfg.gat_in_dim, out_channels=hidden_per_head,
            heads=heads_1, edge_dim=cfg.edge_dim, concat=True,
            dropout=cfg.gat_dropout,
        )
        self.conv2 = GATv2Conv(
            in_channels=cfg.gat_hidden, out_channels=cfg.gat_out_dim,
            heads=cfg.gat_heads, edge_dim=cfg.edge_dim, concat=False,
            dropout=cfg.gat_dropout,
        )
        self.bn1 = nn.BatchNorm1d(cfg.gat_hidden)
        self.bn2 = nn.BatchNorm1d(cfg.gat_out_dim)
        self.dropout = nn.Dropout(cfg.gat_dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        return global_mean_pool(x, batch)


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Attention Is All You Need)."""

    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """Transformer encoder over the sequence of per-timestep graph embeddings."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers, norm=nn.LayerNorm(cfg.d_model)
        )

    def forward(self, x):
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x.mean(dim=1)


class STGTModel(nn.Module):
    """Full Spatial-Temporal Graph Transformer with classification + regression."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.spatial_gat = SpatialGAT(cfg)
        self.temporal_transformer = TemporalTransformer(cfg)

        d = cfg.d_model
        self.shared_head = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, d // 2), nn.LayerNorm(d // 2), nn.GELU(), nn.Dropout(cfg.dropout),
        )
        d_shared = d // 2
        self.cls_head = nn.Linear(d_shared, cfg.n_classes)
        self.reg_head = nn.Linear(d_shared, cfg.n_reg)

    def forward(self, graph_sequences):
        """graph_sequences: list[batch] of list[T] of PyG Data."""
        device = next(self.parameters()).device
        B = len(graph_sequences)
        T = len(graph_sequences[0])

        all_graphs = []
        for seq in graph_sequences:
            all_graphs.extend(seq)

        batched = Batch.from_data_list(all_graphs).to(device)
        gat_out = self.spatial_gat(batched)
        gat_out = gat_out.view(B, T, self.cfg.d_model)

        temporal_out = self.temporal_transformer(gat_out)
        shared = self.shared_head(temporal_out)
        return self.cls_head(shared), self.reg_head(shared)
