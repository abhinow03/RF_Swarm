import torch

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 8 classes total (7 base + 1 transitioning)
FORMATION_NAMES = [
    "v_shape",       # 0
    "encirclement",  # 1
    "column",        # 2
    "diamond",       # 3
    "dispersed",     # 4
    "converging",    # 5
    "shield",        # 6
    "transitioning", # 7
]

CFG = {
    # --- Graph construction ---
    "edge_threshold": 2.0,      # in NORMALISED units. ~50m real distance.

    # --- GAT (spatial) ---
    "gat_in_dim":   3,          # input node features: (x, y, z)
    "gat_hidden":   64,         # hidden dim inside GAT
    "gat_out_dim":  128,        # output node embedding dim
    "gat_heads":    8,          # number of attention heads in GAT
    "gat_dropout":  0.1,

    # --- Temporal Transformer ---
    "d_model":      128,        # must match gat_out_dim
    "n_heads":      8,          # number of self-attention heads
    "n_layers":     4,          # number of transformer encoder layers
    "d_ff":         512,        # feed-forward hidden dim inside transformer
    "dropout":      0.1,
    "max_seq_len":  50,         # number of timesteps

    # --- Output heads ---
    "n_classes":    8,          # 8 formation types (including transitioning)
    "n_reg":        3,          # centroid_velocity, approach_rate, stability

    # --- Training ---
    "batch_size":   64,
    "epochs":       80,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "patience":     12,         # early stopping patience
    "grad_clip":    1.0,        # gradient clipping max norm

    # --- Loss weights ---
    "cls_weight":   1.0,        # weight on classification loss
    "reg_weight":   0.1,        # weight on regression losses (scale separately)
}
