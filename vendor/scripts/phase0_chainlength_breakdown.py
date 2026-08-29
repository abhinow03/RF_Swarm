"""
V5 program, Phase 0, post-guard-fix step 3 (docs/HISTORY.md): before the guard fix,
scripts/phase0_decompose_failures.py found chain-length-2 (single real transition)
accuracy was near zero while chain-length-1 (steady state) succeeded -- a second
possible structural bug distinct from the dispersed_converging guard. Re-measures
that breakdown with the FIXED guard (stgt_bridge.py, 2026-08-09), extends it to
chain-length 3+ (previously excluded -- no RULES key exists, but bucket-A firings on
these are BY DEFINITION wrong-key production, worth counting), and if chain-2 is
still disproportionately broken, traces 20 failing chain-2 trajectories stage by
stage (window classifications -> ambiguity guard -> temporal transition derivation
-> formation_history reduction -> bucket) to find exactly where the correct pair is
lost.

Same seed=999 population, same sample_chain/build_long_sequence_labeled helpers as
scripts/phase0_decompose_failures.py (imported, not duplicated) -- reproduces the
identical 1000 trajectories index-for-index as every other Phase 0 measurement.

Usage (run inside tmux):
    python scripts/phase0_chainlength_breakdown.py --n 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.coverage import classify_observation, BUCKET_A  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from phase0_decompose_failures import (  # noqa: E402
    sample_chain, build_long_sequence_labeled, ground_truth_pair, position_bucket, CLASS_ORDER,
)
from build_sft_dataset import RULES  # noqa: E402

DATA_DIR = REPO / "swarm_data"
CHECKPOINT = DATA_DIR / "best_model.pt"
SEED = 999
N_TRACE = 20


def chain_length_bucket(chain: list) -> str:
    if len(chain) == 1:
        return "1"
    if len(chain) == 2:
        return "2"
    return "3+"


def score_pair_level(records, key: str):
    """records: list of dicts with gt_pair/recovered_pair (or None). Returns
    n, n_correct, accuracy, plus threat/intent/action accuracy via RULES (no
    recovery scores as wrong on every metric, same convention as
    phase0_threat_ceiling.py)."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    n_pair_correct = threat_correct = intent_correct = action_correct = 0
    for r in records:
        gt_pair = tuple(r["gt_pair"])
        true_threat, true_intent, true_action = RULES[gt_pair]
        rec = r[key]
        if rec is None:
            continue
        rec_pair = tuple(rec)
        n_pair_correct += rec_pair == gt_pair
        if rec_pair in RULES:
            rec_threat, rec_intent, rec_action = RULES[rec_pair]
            threat_correct += rec_threat == true_threat
            intent_correct += rec_intent == true_intent
            action_correct += rec_action == true_action
    return {"n": n, "n_pair_correct": n_pair_correct, "pair_accuracy": n_pair_correct / n,
            "threat_accuracy": threat_correct / n, "intent_accuracy": intent_correct / n,
            "action_accuracy": action_correct / n}


def trace_trajectory(idx, chain, gt_pair, predictions, true_labels, window_size=50, stride=10):
    lines = [f"=== trajectory {idx}: true_chain={chain}, gt_pair={gt_pair} ==="]

    window_true = []
    for w, pred in enumerate(predictions):
        start, end = w * stride, min(w * stride + window_size, len(true_labels))
        window_true_labels = true_labels[start:end]
        true_label = Counter(window_true_labels).most_common(1)[0][0] if window_true_labels else None
        window_true.append(true_label)

    lines.append(f"n_windows={len(predictions)}")
    lines.append(f"{'idx':>4} {'true':>13} {'pred':>13} {'conf':>6} {'top2':>30} {'d/c_amb':>8}")
    from swarm_intent.stgt_bridge import _is_ambiguous_dispersed_converging
    for w, (pred, true_label) in enumerate(zip(predictions, window_true)):
        cp = pred.get("class_probabilities", {})
        ranked = sorted(cp.items(), key=lambda kv: kv[1], reverse=True)[:2]
        top2_str = ",".join(f"{n}={p:.2f}" for n, p in ranked)
        amb = _is_ambiguous_dispersed_converging(cp)
        mark = " <-WRONG" if true_label != pred["formation_type"] else ""
        lines.append(f"{w:>4} {str(true_label):>13} {pred['formation_type']:>13} "
                     f"{pred['formation_confidence']:>6.2f} {top2_str:>30} {str(amb):>8}{mark}")

    bucket_info = classify_observation(predictions, robust=False)
    summary = bucket_info["summary"]
    lines.append(f"\nformation_history (raw, consecutive-collapsed): {summary.get('formation_history')}")
    lines.append(f"transitions_detected (temporal): {summary.get('transitions_detected')}")
    lines.append(f"n_unknown_windows={summary.get('n_unknown_windows')}, "
                 f"n_ambiguous_dispersed_converging_windows={summary.get('n_ambiguous_dispersed_converging_windows')}")
    lines.append(f"FINAL: bucket={bucket_info['bucket']}, subtype={bucket_info['subtype']}, "
                 f"guard_reasons={bucket_info['guard_reasons']}, rules_key={bucket_info['rules_key']}")

    recovered = tuple(bucket_info["rules_key"]) if bucket_info["bucket"] == BUCKET_A else None
    correct = recovered == gt_pair

    # diagnose WHERE the pair was lost
    n_wrong_windows = sum(1 for t, p in zip(window_true, predictions) if t != p["formation_type"])
    if correct:
        diagnosis = "CORRECT"
    elif bucket_info["bucket"] == BUCKET_A:
        diagnosis = "structural_reduction_wrong_pair"  # reached A, but wrong pair
    elif "dispersed_converging_ambiguity" in bucket_info["guard_reasons"]:
        diagnosis = "blocked_by_ambiguity_guard"
    elif bucket_info["subtype"] == "terminal_unknown":
        diagnosis = "trailing_transitioning_run"
    elif bucket_info["subtype"] == "all_unknown":
        diagnosis = "all_windows_transitioning"
    elif bucket_info["subtype"] in ("multi_hop", "oscillation"):
        diagnosis = "spurious_third_formation_from_misclassification" if n_wrong_windows else "unexplained_multi_hop"
    elif "oov_name" in bucket_info["guard_reasons"]:
        diagnosis = "blocked_by_oov_name_guard"
    elif "dominant_history_contradiction" in bucket_info["guard_reasons"]:
        diagnosis = "blocked_by_dominant_tie_guard"
    else:
        diagnosis = "other"

    lines.append(f"n_wrong_windows={n_wrong_windows}/{len(predictions)}, DIAGNOSIS: {diagnosis}")
    return "\n".join(lines), diagnosis


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
    reporter = Reporter("phase0_chainlength_breakdown", args.n, rate_hint=8.0)

    records = {"1": [], "2": [], "3+": []}
    chain2_failures = []  # (idx, chain, gt_pair, predictions, true_labels) for tracing
    bucket_3plus = Counter()

    for i in range(args.n):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq, true_labels = build_long_sequence_labeled(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)

        cl = chain_length_bucket(chain)
        gt_pair = ground_truth_pair(chain)

        info_false = classify_observation(predictions, robust=False)
        info_true = classify_observation(predictions, robust=True)
        rec_false = list(info_false["rules_key"]) if info_false["bucket"] == BUCKET_A else None
        rec_true = list(info_true["rules_key"]) if info_true["bucket"] == BUCKET_A else None

        if cl == "3+":
            bucket_3plus[info_false["bucket"]] += 1
            bucket_3plus[f"{info_false['bucket']}_robust"] += 1 if info_true["bucket"] == BUCKET_A else 0
        else:
            rec = {"i": i, "gt_pair": list(gt_pair), "recovered_pair": rec_false,
                  "recovered_pair_robust": rec_true}
            records[cl].append(rec)
            if cl == "2" and rec_false != list(gt_pair) and len(chain2_failures) < 200:
                chain2_failures.append((i, chain, gt_pair, predictions, true_labels))

        reporter.update(1, item=f"seq {i}")

    reporter.status = "done"
    reporter._write()

    print("\n" + "=" * 100)
    print("PAIR-LEVEL AND THREAT CEILING BY CHAIN LENGTH (robust=False / robust=True)")
    print("=" * 100)
    print("| chain_length | n | pair_acc (F) | pair_acc (T) | threat_acc (F) | threat_acc (T) |")
    print("|---|---|---|---|---|---|")
    summary_out = {}
    for cl in ("1", "2"):
        s_false = score_pair_level(records[cl], "recovered_pair")
        s_true = score_pair_level(records[cl], "recovered_pair_robust")
        summary_out[cl] = {"robust_false": s_false, "robust_true": s_true}
        print(f"| {cl} | {s_false['n']} | {s_false['pair_accuracy']:.1%} | {s_true['pair_accuracy']:.1%} | "
             f"{s_false['threat_accuracy']:.1%} | {s_true['threat_accuracy']:.1%} |")

    n_3plus = sum(v for k, v in bucket_3plus.items() if not k.endswith("_robust"))
    a_3plus = bucket_3plus.get(BUCKET_A, 0)
    a_3plus_robust = bucket_3plus.get(f"{BUCKET_A}_robust", 0)
    print(f"| 3+ | {n_3plus} | n/a (no RULES key) | n/a | n/a | n/a |")
    print(f"\nchain 3+ FALSE-POSITIVE bucket-A rate (any A-resolution here is wrong by "
         f"construction -- no 2-tuple ground truth exists): "
         f"robust=False {a_3plus}/{n_3plus} ({a_3plus/n_3plus:.1%}), "
         f"robust=True {a_3plus_robust}/{n_3plus} ({a_3plus_robust/n_3plus:.1%})")
    summary_out["3+"] = {"n": n_3plus, "bucket_A_false_positive_rate_robust_false": a_3plus / n_3plus,
                         "bucket_A_false_positive_rate_robust_true": a_3plus_robust / n_3plus,
                         "bucket_distribution": dict(bucket_3plus)}

    verdict = ("STILL BROKEN" if summary_out["2"]["robust_false"]["pair_accuracy"] < 0.15
              else "recovered, not a second structural bug")
    print(f"\nVERDICT: chain-length-2 pair accuracy (robust=False) = "
         f"{summary_out['2']['robust_false']['pair_accuracy']:.1%} -- {verdict}")

    (REPO / "evaluation" / "phase0_chainlength_breakdown.json").write_text(json.dumps(summary_out, indent=2))
    print(f"\nsaved evaluation/phase0_chainlength_breakdown.json")

    # ---- stage trace for up to N_TRACE failing chain-2 trajectories ----
    trace_sample = chain2_failures[:N_TRACE]
    print(f"\n\n{'='*100}\nSTAGE TRACE: {len(trace_sample)} failing chain-length-2 trajectories\n{'='*100}")
    trace_texts = []
    diag_counts = Counter()
    for idx, chain, gt_pair, predictions, true_labels in trace_sample:
        text, diagnosis = trace_trajectory(idx, chain, gt_pair, predictions, true_labels)
        trace_texts.append(text)
        diag_counts[diagnosis] += 1

    trace_path = REPO / "evaluation" / "phase0_chain2_trace.txt"
    trace_path.write_text("\n\n".join(trace_texts))
    print(f"saved full trace to {trace_path}")

    print(f"\ndiagnosis distribution across {len(trace_sample)} traced failures:")
    for diag, c in diag_counts.most_common():
        print(f"  {diag}: {c}/{len(trace_sample)} ({c/len(trace_sample):.1%})")


if __name__ == "__main__":
    main()
