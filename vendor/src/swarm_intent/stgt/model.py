import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
import math
from .config import CFG, device

# =============================================================================
# PHASE A — GRAPH CONSTRUCTION
# =============================================================================

def build_graph(positions, threshold=None):
    if threshold is None:
        threshold = CFG["edge_threshold"]

    if isinstance(positions, np.ndarray):
        positions = torch.tensor(positions, dtype=torch.float32)

    n = positions.shape[0]

    src_list, dst_list, attr_list = [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[j] - positions[i]
            dist = torch.norm(diff).item()

            if dist < threshold:
                src_list.append(i)
                dst_list.append(j)
                attr_list.append(torch.cat([diff, torch.tensor([dist])]))

                src_list.append(j)
                dst_list.append(i)
                attr_list.append(torch.cat([-diff, torch.tensor([dist])]))

    if len(src_list) == 0:
        src_list = list(range(n))
        dst_list = list(range(n))
        attr_list = [torch.zeros(4) for _ in range(n)]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.stack(attr_list, dim=0)

    # Centroid-relative node features, not absolute positions -- kept in sync
    # with src/swarm_intent/graph.py's build_graph. See that function's
    # docstring and V5_LOG.md (2026-08-08) for why.
    centroid = positions.mean(dim=0, keepdim=True)
    node_features = positions - centroid
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


def sequence_to_graphs(seq, threshold=None):
    return [build_graph(seq[t], threshold) for t in range(seq.shape[0])]


# =============================================================================
# PHASE B — SPATIAL GAT LAYER
# =============================================================================

class SpatialGAT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        heads_1 = cfg["gat_heads"]
        hidden_per_head = cfg["gat_hidden"] // heads_1

        self.conv1 = GATv2Conv(
            in_channels=cfg["gat_in_dim"],
            out_channels=hidden_per_head,
            heads=heads_1,
            edge_dim=4,
            concat=True,
            dropout=cfg["gat_dropout"],
        )
        self.conv2 = GATv2Conv(
            in_channels=cfg["gat_hidden"],
            out_channels=cfg["gat_out_dim"],
            heads=cfg["gat_heads"],
            edge_dim=4,
            concat=False,
            dropout=cfg["gat_dropout"],
        )
        self.bn1 = nn.BatchNorm1d(cfg["gat_hidden"])
        self.bn2 = nn.BatchNorm1d(cfg["gat_out_dim"])
        self.dropout = nn.Dropout(cfg["gat_dropout"])

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

        out = global_mean_pool(x, batch)
        return out


# =============================================================================
# PHASE C — TEMPORAL TRANSFORMER
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pos_enc = PositionalEncoding(
            d_model=cfg["d_model"],
            max_len=cfg["max_seq_len"],
            dropout=cfg["dropout"],
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["d_model"],
            nhead=cfg["n_heads"],
            dim_feedforward=cfg["d_ff"],
            dropout=cfg["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg["n_layers"],
            norm=nn.LayerNorm(cfg["d_model"]),
        )

    def forward(self, x):
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return x


# =============================================================================
# PHASE D — OUTPUT HEADS
# =============================================================================

class STGTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.spatial_gat = SpatialGAT(cfg)
        self.temporal_transformer = TemporalTransformer(cfg)

        d = cfg["d_model"]
        self.shared_head = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(d, d // 2),
            nn.LayerNorm(d // 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
        )
        d_shared = d // 2

        self.cls_head = nn.Linear(d_shared, cfg["n_classes"])
        self.reg_head = nn.Linear(d_shared, cfg["n_reg"])

    def forward(self, graph_sequences):
        B = len(graph_sequences)
        T = len(graph_sequences[0])

        all_graphs = []
        for seq in graph_sequences:
            all_graphs.extend(seq)

        batched = Batch.from_data_list(all_graphs).to(device)
        gat_out = self.spatial_gat(batched)
        gat_out = gat_out.view(B, T, self.cfg["d_model"])

        temporal_out = self.temporal_transformer(gat_out)
        shared = self.shared_head(temporal_out)

        logits  = self.cls_head(shared)
        reg_out = self.reg_head(shared)

        return logits, reg_out
