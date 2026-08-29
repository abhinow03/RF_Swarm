"""
Inference: single-window prediction, sliding-window over a long stream, and
assembly of the tactical-context string + LLM prompt.

Merges the old ``predict`` / ``predict_v2`` into one ``predict`` that takes the
active ``formation_names`` (so 7- and 8-class models share one code path).

CAVEAT: the transition_from / transition_to derivation here is reconstructed
(top-2 non-transition classes by probability). Verify against your original
``predict_v2`` if you depend on exact values.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from . import context_spec as spec
from .calibration import AbsoluteCalibrator, Calibrator
from .config import Config, TRANSITION_CLASS
from .graph import sequence_to_graphs


def infer_behavior_trend(reg_preds, cls_probs, formation_names) -> str:
    _, approach_rate, stability = reg_preds
    order = np.argsort(cls_probs)[::-1]
    trend_formation = formation_names[order[1]]
    if approach_rate < -0.05:
        return "converging"
    if approach_rate > 0.05:
        return "dispersing"
    if stability < 0.3:
        return f"transitioning_to_{trend_formation}"
    return "stable"


@torch.no_grad()
def predict(model, sequence, cfg: Config, reg_mean, reg_std, formation_names):
    """Run one NORMALISED (50,6,3) window -> structured dict for the LLM."""
    model.eval()
    graphs = sequence_to_graphs(sequence, cfg.edge_threshold)
    logits, reg_out = model([graphs])

    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(probs.argmax())
    confidence = float(probs.max())

    reg_real = reg_out.cpu().numpy()[0] * reg_std + reg_mean
    centroid_velocity = float(reg_real[0])
    approach_rate = float(reg_real[1])
    formation_stability = float(np.clip(reg_real[2], 0.0, 1.0))

    centroids = sequence.mean(axis=1)
    centered = sequence - centroids[:, None, :]
    drone_mean_dist = np.linalg.norm(centered, axis=2).mean(axis=0)
    role_differentiation = bool(drone_mean_dist.max() > 2.0 * np.median(drone_mean_dist))

    formation = formation_names[pred_class]
    transition_from = transition_to = None
    if formation == TRANSITION_CLASS:
        non_trans = [(formation_names[i], probs[i]) for i in range(len(formation_names))
                     if formation_names[i] != TRANSITION_CLASS]
        non_trans.sort(key=lambda x: x[1], reverse=True)
        transition_to = non_trans[0][0]
        transition_from = non_trans[1][0]

    return {
        "formation_type": formation,
        "formation_confidence": round(confidence, 4),
        "centroid_velocity": round(centroid_velocity, 3),
        "approach_rate": round(approach_rate, 3),
        "formation_stability": round(formation_stability, 4),
        "role_differentiation": role_differentiation,
        "transition_from": transition_from,
        "transition_to": transition_to,
        "behavior_trend": infer_behavior_trend(reg_real, probs, formation_names),
        "class_probabilities": {
            formation_names[i]: round(float(probs[i]), 4) for i in range(len(formation_names))
        },
    }


def sliding_window_inference(model, long_sequence, cfg, reg_mean, reg_std,
                             train_mean, train_std, formation_names,
                             window_size=50, stride=10, dt=0.5):
    """Slide over a RAW (T,6,3) stream; normalise each window with TRAIN stats."""
    T = long_sequence.shape[0]
    assert T >= window_size, f"Sequence too short: {T} < {window_size}"
    predictions = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        window_norm = (long_sequence[start:end] - train_mean) / train_std
        pred = predict(model, window_norm, cfg, reg_mean, reg_std, formation_names)
        pred.update({"window_start_t": int(start), "window_end_t": int(end - 1),
                     "time_start_s": round(start * dt, 1), "time_end_s": round((end - 1) * dt, 1)})
        predictions.append(pred)
    return predictions


def build_tactical_context(predictions, calibrator: "Calibrator | None" = None):
    """Summarise a chain of window predictions into (context_str, summary_dict).

    calibrator: maps raw STGT scalars to the frozen narrative vocabulary in
    context_spec.py (see calibration.py). Defaults to AbsoluteCalibrator() — today's
    hardcoded cutoffs — so every existing caller sees byte-identical output. Pass a
    fitted PercentileCalibrator once the STGT retrain changes units.
    """
    calibrator = calibrator or AbsoluteCalibrator()
    n = len(predictions)
    if n == 0:
        return "No predictions available.", {}

    formation_seq = [p["formation_type"] for p in predictions]
    transitions = [
        {"at_time_s": predictions[i]["time_start_s"], "from": formation_seq[i - 1], "to": formation_seq[i]}
        for i in range(1, n) if formation_seq[i] != formation_seq[i - 1]
    ]
    non_trans = [f for f in formation_seq if f != TRANSITION_CLASS]
    dominant = max(set(non_trans), key=non_trans.count) if non_trans else TRANSITION_CLASS

    velocities = [p["centroid_velocity"] for p in predictions]
    mid = n // 2
    delta_v = (np.mean(velocities[mid:]) - np.mean(velocities[:mid])) if mid > 0 else 0.0
    vel_trend = calibrator.velocity_trend(float(delta_v))

    stabilities = [p["formation_stability"] for p in predictions]
    mean_stability = float(np.mean(stabilities))
    early, late = np.mean(stabilities[:max(1, mid)]), (np.mean(stabilities[mid:]) if mid > 0 else stabilities[-1])
    stab_trend = calibrator.stability_trend(float(early), float(late))

    approach = [p["approach_rate"] for p in predictions]
    mean_approach = float(np.mean(approach))
    approach_summary = calibrator.spread_dynamics(mean_approach)

    role_true_count = sum(1 for p in predictions if p["role_differentiation"])
    role_str = calibrator.role_differentiation(role_true_count, n)
    role_flag = (role_str == spec.ROLE_DIFFERENTIATION_PRESENT)  # summary dict keeps its bool shape
    confidences = [p["formation_confidence"] for p in predictions]
    mean_conf, low_conf = float(np.mean(confidences)), sum(1 for c in confidences if c < 0.6)

    lines = [
        f"Observation window: {predictions[0]['time_start_s']}s - {predictions[-1]['time_end_s']}s "
        f"({n} overlapping {len(predictions) and 50}-step windows)",
        f"Dominant formation: {dominant}",
        f"Formation history: {' -> '.join(dict.fromkeys(formation_seq))}",
    ]
    lines += ([f"Transition at t={t['at_time_s']}s: {t['from']} -> {t['to']}" for t in transitions]
              or [spec.NO_TRANSITIONS_DETECTED])
    lines += [
        f"Velocity trend: {vel_trend} (delta_v={delta_v:+.2f})",
        f"Formation stability: {stab_trend} (mean={mean_stability:.2f})",
        f"Spread dynamics: {approach_summary} (mean approach_rate={mean_approach:.3f})",
        f"Role differentiation: {role_str}",
        spec.CONFIDENCE_LINE_TEMPLATE.format(mean_conf=mean_conf, low_conf=low_conf),
    ]
    summary = {
        "dominant_formation": dominant, "formation_history": list(dict.fromkeys(formation_seq)),
        "transitions_detected": transitions, "velocity_trend": vel_trend,
        "delta_velocity": round(float(delta_v), 3), "stability_trend": stab_trend,
        "mean_stability": round(mean_stability, 3), "spread_dynamics": approach_summary,
        "mean_approach_rate": round(mean_approach, 3), "role_differentiation": role_flag,
        "mean_confidence": round(mean_conf, 3), "low_conf_windows": low_conf, "n_windows": n,
    }
    return "\n".join(lines), summary


# Output schema the LLM must return. Single source of truth for prompts + eval.
OUTPUT_SCHEMA = {
    "situation_summary": "<2-3 sentence plain-English description>",
    "threat_level": "<low / medium / high / critical / unknown>",
    "threat_reasoning": "<why this threat level>",
    "likely_intent": "<surveillance / approach / encircle / patrol / defensive / withdraw / "
                     "regroup / consolidate / transit / defensive_transition / area_search / "
                     "attack_preparation / rally / reposition / unknown>",
    "recommended_action": "<monitor / increase_surveillance / alert_operator / deploy_countermeasure / intercept>",
    "confidence_in_assessment": "<low / medium / high>",
    "key_indicators": ["<indicator 1>", "<indicator 2>", "<indicator 3>"],
    "follow_up_watch": "<what to monitor next window>",
}


def build_llm_prompt(predictions, tactical_context, summary):
    """Assemble the operator-analyst prompt sent to the LLM."""
    key_windows = []
    for p in predictions:
        if (p is predictions[0] or p is predictions[-1]
                or p["formation_type"] == TRANSITION_CLASS or p["formation_confidence"] < 0.65):
            key_windows.append({
                "t": f"{p['time_start_s']}-{p['time_end_s']}s", "formation": p["formation_type"],
                "confidence": p["formation_confidence"], "velocity": p["centroid_velocity"],
                "approach": p["approach_rate"], "stability": p["formation_stability"],
                "from": p.get("transition_from"), "to": p.get("transition_to"),
            })
    return f"""You are a tactical AI analyst for a counter-UAV system. Interpret the \
structured sensor outputs below and give a concise tactical assessment for a human operator.

Output a JSON object with EXACTLY these fields:
{json.dumps(OUTPUT_SCHEMA, indent=2)}

--- TACTICAL CONTEXT ---
{tactical_context}

--- KEY WINDOW PREDICTIONS ---
{json.dumps(key_windows, indent=2)}

Respond with ONLY the JSON object. No preamble, no text outside the JSON."""
