import numpy as np
import torch
import torch.nn.functional as F
from .config import FORMATION_NAMES
from .model import sequence_to_graphs


@torch.no_grad()
def predict_v2(model, sequence, cfg, reg_mean, reg_std, formation_names=None):
    if formation_names is None:
        formation_names = FORMATION_NAMES

    model.eval()
    graphs  = sequence_to_graphs(sequence, threshold=cfg["edge_threshold"])
    logits, reg_out = model([graphs])

    probs      = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(probs.argmax())
    confidence = float(probs.max())

    reg_raw  = reg_out.cpu().numpy()[0]
    reg_real = reg_raw * reg_std + reg_mean

    centroid_velocity   = float(reg_real[0])
    approach_rate       = float(reg_real[1])
    formation_stability = float(np.clip(reg_real[2], 0.0, 1.0))

    centroids       = sequence.mean(axis=1)
    centered        = sequence - centroids[:, None, :]
    per_drone_dist  = np.linalg.norm(centered, axis=2)
    drone_mean_dist = per_drone_dist.mean(axis=0)
    role_diff       = bool(drone_mean_dist.max() > 2.0 * np.median(drone_mean_dist))

    transition_from = None
    transition_to   = None
    if pred_class == 7:   # "transitioning"
        non_trans_probs = probs.copy()
        non_trans_probs[7] = -1
        sorted_idx      = np.argsort(non_trans_probs)[::-1]
        transition_from = formation_names[sorted_idx[0]]
        transition_to   = formation_names[sorted_idx[1]]

    output = {
        "formation_type":       formation_names[pred_class],
        "formation_confidence": round(confidence, 4),
        "centroid_velocity":    round(centroid_velocity, 3),
        "approach_rate":        round(approach_rate, 3),
        "formation_stability":  round(formation_stability, 4),
        "role_differentiation": role_diff,
        "transition_from":      transition_from,
        "transition_to":        transition_to,
        "class_probabilities":  {
            formation_names[i]: round(float(probs[i]), 4)
            for i in range(len(formation_names))
        },
    }
    return output

def sliding_window_inference(
    model,
    long_sequence,
    cfg,
    reg_mean,
    reg_std,
    train_mean,
    train_std,
    window_size=50,
    stride=10,
    dt=0.5,
    formation_names=None,
):
    if formation_names is None:
        formation_names = FORMATION_NAMES

    T = long_sequence.shape[0]
    assert T >= window_size, f"Sequence too short: {T} < {window_size}"

    predictions = []
    for start in range(0, T - window_size + 1, stride):
        end    = start + window_size
        window = long_sequence[start:end]
        window_norm = (window - train_mean) / train_std
        pred = predict_v2(model, window_norm, cfg, reg_mean, reg_std, formation_names)

        pred["window_start_t"] = int(start)
        pred["window_end_t"]   = int(end - 1)
        pred["time_start_s"]   = round(start * dt, 1)
        pred["time_end_s"]     = round((end - 1) * dt, 1)
        predictions.append(pred)

    return predictions
