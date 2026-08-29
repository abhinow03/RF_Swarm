"""
Step 3 of the 2026-08-10 full-scale go/no-go (docs/V5_LOG.md step 52): the real gate. Runs
the EXACT SAME stratified ceiling protocol used throughout Phase 0
(scripts/phase0_chainlength_breakdown.py: seed=999, n=1000, chain-length stratification,
robust=False/True both reported) against TWO checkpoints side by side:
  - the standing baseline: swarm_data/best_model.pt (strategy-5 checkpoint, guard fixes +
    dwell-time fix + source symmetrization already applied in stgt_bridge.py/coverage.py --
    those are code-level fixes, apply identically regardless of which checkpoint is scored)
  - the full-scale candidate: swarm_data_candidate_fullscale/best_model.pt (corrected-port,
    n_transition=9000 hops matching baseline's independent-sample count, robust=True
    normalization, uncapped acceleration -- docs/V5_LOG.md steps 50-51)

Not a new methodology -- parameterizes phase0_chainlength_breakdown.py's own logic
(sample_chain/build_long_sequence_labeled/ground_truth_pair/classify_observation/RULES) by
checkpoint+data_dir so both can be scored on the identical population without duplicating or
drifting from the established protocol.

Usage (run inside tmux):
    python scripts/phase0_fullscale_ceiling_comparison.py --n 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.coverage import classify_observation, BUCKET_A  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402
from swarm_intent.stgt.model import STGTModel  # noqa: E402
from swarm_intent.stgt.inference import sliding_window_inference  # noqa: E402
from swarm_intent.eval_trajectories import sample_chain, build_long_sequence_labeled, ground_truth_pair  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402

SEED = 999


def chain_length_bucket(chain):
    if len(chain) == 1:
        return "1"
    if len(chain) == 2:
        return "2"
    return "3+"


def score_pair_level(records, key):
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


def run_ceiling(checkpoint_path, data_dir, n, seed, label, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"[{label}] checkpoint: epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f}")
    model = STGTModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    train_mean = np.load(data_dir / "train_mean.npy")
    train_std = np.load(data_dir / "train_std.npy")
    reg_mean = ckpt["reg_mean"]
    reg_std = ckpt["reg_std"]

    rng = np.random.default_rng(seed)
    reporter = Reporter(f"fullscale_ceiling_{label}", n, rate_hint=8.0)
    records = {"1": [], "2": [], "3+": []}
    bucket_3plus = Counter()

    for i in range(n):
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
            records[cl].append({"i": i, "gt_pair": list(gt_pair), "recovered_pair": rec_false,
                               "recovered_pair_robust": rec_true})
        reporter.update(1, item=f"seq {i}")

    reporter.status = "done"
    reporter._write()

    summary = {}
    for cl in ("1", "2"):
        s_false = score_pair_level(records[cl], "recovered_pair")
        s_true = score_pair_level(records[cl], "recovered_pair_robust")
        summary[cl] = {"robust_false": s_false, "robust_true": s_true}

    n_3plus = sum(v for k, v in bucket_3plus.items() if not k.endswith("_robust"))
    a_3plus = bucket_3plus.get(BUCKET_A, 0)
    a_3plus_robust = bucket_3plus.get(f"{BUCKET_A}_robust", 0)
    summary["3+"] = {"n": n_3plus,
                     "bucket_A_false_positive_rate_robust_false": a_3plus / n_3plus if n_3plus else None,
                     "bucket_A_false_positive_rate_robust_true": a_3plus_robust / n_3plus if n_3plus else None}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_baseline = run_ceiling(REPO / "swarm_data" / "best_model.pt", REPO / "swarm_data",
                                   args.n, args.seed, "baseline", device)
    summary_candidate = run_ceiling(REPO / "swarm_data_candidate_fullscale" / "best_model.pt",
                                    REPO / "swarm_data_candidate_fullscale",
                                    args.n, args.seed, "candidate", device)

    print("\n" + "=" * 100)
    print(f"FULL CEILING COMPARISON, n={args.n}, seed={args.seed} -- THREAT terms first, robust=False / robust=True")
    print("=" * 100)
    print("| chain | n_b | threat_b(F) | threat_b(T) | pair_b(F) | pair_b(T) | n_c | threat_c(F) | threat_c(T) | pair_c(F) | pair_c(T) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cl in ("1", "2"):
        b, c = summary_baseline[cl], summary_candidate[cl]
        print(f"| {cl} | {b['robust_false']['n']} | {b['robust_false']['threat_accuracy']:.1%} | "
             f"{b['robust_true']['threat_accuracy']:.1%} | {b['robust_false']['pair_accuracy']:.1%} | "
             f"{b['robust_true']['pair_accuracy']:.1%} | {c['robust_false']['n']} | "
             f"{c['robust_false']['threat_accuracy']:.1%} | {c['robust_true']['threat_accuracy']:.1%} | "
             f"{c['robust_false']['pair_accuracy']:.1%} | {c['robust_true']['pair_accuracy']:.1%} |")

    print(f"\nchain 3+ bucket-A false-positive rate: baseline robust=F "
         f"{summary_baseline['3+']['bucket_A_false_positive_rate_robust_false']:.1%}, robust=T "
         f"{summary_baseline['3+']['bucket_A_false_positive_rate_robust_true']:.1%}  |  candidate robust=F "
         f"{summary_candidate['3+']['bucket_A_false_positive_rate_robust_false']:.1%}, robust=T "
         f"{summary_candidate['3+']['bucket_A_false_positive_rate_robust_true']:.1%}")

    out = {"baseline": summary_baseline, "candidate": summary_candidate}
    (REPO / "evaluation" / "phase0_fullscale_ceiling_comparison.json").write_text(json.dumps(out, indent=2))
    print("\nsaved evaluation/phase0_fullscale_ceiling_comparison.json")


if __name__ == "__main__":
    main()
