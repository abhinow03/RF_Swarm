"""
AUDIT.md sec AM follow-up, step 1: reproduce the headline ceiling figures (83.0% threat /
77.3% pair pooled; chain-1 87.6%/85.8%; chain-2 77.2%/66.7%) as a REAL, persisted artifact by
scoring the LOCKED population (eval_data/LOCKED_seed999_FINAL.json) directly -- no
regeneration, no resampling. Verifies the locked file's sha256 first (integrity gate), then
runs the exact same STGT+bridge scoring phase0_ceiling.py/phase0_threat_ceiling.py used
(sliding_window_inference -> classify_observation -> RULES), stratified by chain length per
CEILING.md's own 2026-08-09 stratification policy.

No training, no corpus changes -- CPU inference only against the already-locked, frozen
strategy-5 checkpoint (swarm_data/best_model.pt, hash-verified unchanged throughout this
project's history).

Usage:
    python scripts/rule0_am_reproduce_headline.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.coverage import classify_observation, BUCKET_A  # noqa: E402
from swarm_intent.eval_trajectories import ground_truth_pair  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402

LOCKED_PATH = REPO / "eval_data" / "LOCKED_seed999_FINAL.json"
CITED_SHA = "871a9dae4c6fdf08e1aed803592fa7c61b1a852c150693b5819fe2271717b96e"
DATA_DIR = REPO / "swarm_data"
CHECKPOINT = DATA_DIR / "best_model.pt"

# Headline figures currently cited in prose (docs/CEILING.md "Current state, 2026-08-10" /
# "Current state (stratified), 2026-08-09"), to be checked against, NOT silently adjusted.
CITED = {
    "pooled": {"threat": 0.830, "pair": 0.773},
    "chain_1": {"threat": 0.876, "pair": 0.858},
    "chain_2": {"threat": 0.772, "pair": 0.667},
}


def main():
    raw = LOCKED_PATH.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    assert actual_sha == CITED_SHA, (
        f"INTEGRITY FAILURE: eval_data/LOCKED_seed999_FINAL.json sha256 {actual_sha} does not "
        f"match the cited {CITED_SHA} -- refusing to reproduce a headline figure against a "
        f"file that may have drifted since it was locked (AUDIT.md sec AM).")
    print(f"integrity gate: locked file sha256 matches cited hash ({actual_sha[:16]}...) -- OK")

    data = json.loads(raw)
    records = data["records"]
    assert data["seed"] == 999 and data["n"] == 1000 and len(records) == 1000

    import torch
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference
    import swarm_intent.stgt.model as stgt_model_module
    from swarm_intent.stgt.config import device as _unused  # noqa: F401

    device = torch.device("cpu")
    stgt_model_module.device = device
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = STGTModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    per_record = []
    for rec in records:
        chain = rec["chain"]
        gt_pair = ground_truth_pair(chain)
        if gt_pair is None:
            continue  # chain length 3+, not pair-eligible -- same convention as phase0_ceiling.py
        long_seq = np.array(rec["positions"], dtype=np.float64)
        predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        bucket_f = classify_observation(predictions, robust=False)
        bucket_t = classify_observation(predictions, robust=True)
        rec_pair_f = tuple(bucket_f["rules_key"]) if bucket_f["bucket"] == BUCKET_A else None
        rec_pair_t = tuple(bucket_t["rules_key"]) if bucket_t["bucket"] == BUCKET_A else None
        per_record.append({
            "i": rec["i"], "chain_length": len(chain), "gt_pair": list(gt_pair),
            "recovered_pair": list(rec_pair_f) if rec_pair_f else None,
            "recovered_pair_robust": list(rec_pair_t) if rec_pair_t else None,
        })
        if len(per_record) % 100 == 0:
            print(f"  scored {len(per_record)} pair-eligible records...")

    print(f"total pair-eligible: {len(per_record)}")

    def score(recs, pair_field):
        n = len(recs)
        pair_correct = threat_correct = 0
        for r in recs:
            gt = tuple(r["gt_pair"])
            true_threat = RULES[gt][0]
            rec_pair = r[pair_field]
            if rec_pair is None:
                continue
            rec_pair = tuple(rec_pair)
            if rec_pair == gt:
                pair_correct += 1
            if RULES[rec_pair][0] == true_threat:
                threat_correct += 1
        return {"n": n, "pair_accuracy": pair_correct / n if n else None,
               "threat_accuracy": threat_correct / n if n else None}

    strata = {
        "pooled": per_record,
        "chain_1": [r for r in per_record if r["chain_length"] == 1],
        "chain_2": [r for r in per_record if r["chain_length"] == 2],
    }

    results = {}
    for stratum_name, recs in strata.items():
        robust_true = score(recs, "recovered_pair_robust")
        robust_false = score(recs, "recovered_pair")
        results[stratum_name] = {"robust_True": robust_true, "robust_False": robust_false}

    print("\n" + "=" * 100)
    print("REPRODUCED HEADLINE FIGURES vs CITED (robust=True, the value CEILING.md cites)")
    print("=" * 100)
    discrepancies = []
    for stratum_name in ("pooled", "chain_1", "chain_2"):
        actual = results[stratum_name]["robust_True"]
        cited = CITED[stratum_name]
        threat_match = actual["threat_accuracy"] is not None and abs(actual["threat_accuracy"] - cited["threat"]) < 0.0005
        pair_match = actual["pair_accuracy"] is not None and abs(actual["pair_accuracy"] - cited["pair"]) < 0.0005
        print(f"{stratum_name}: n={actual['n']}  "
             f"threat actual={actual['threat_accuracy']:.1%} cited={cited['threat']:.1%} "
             f"{'MATCH' if threat_match else 'DISCREPANCY'}  |  "
             f"pair actual={actual['pair_accuracy']:.1%} cited={cited['pair']:.1%} "
             f"{'MATCH' if pair_match else 'DISCREPANCY'}")
        if not threat_match:
            discrepancies.append({"stratum": stratum_name, "metric": "threat_accuracy",
                                  "actual": actual["threat_accuracy"], "cited": cited["threat"]})
        if not pair_match:
            discrepancies.append({"stratum": stratum_name, "metric": "pair_accuracy",
                                  "actual": actual["pair_accuracy"], "cited": cited["pair"]})

    out = {
        "locked_file": str(LOCKED_PATH.relative_to(REPO)), "locked_file_sha256": actual_sha,
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "n_pair_eligible": len(per_record), "strata": results, "cited": CITED,
        "discrepancies": discrepancies,
        "per_record": per_record,
    }
    out_path = REPO / "evaluation" / "rule0_am_headline_reproduction.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")
    if discrepancies:
        print(f"\n{len(discrepancies)} DISCREPANCIES FOUND -- reporting, not silently adjusting cited figures.")
    else:
        print("\nALL FIGURES REPRODUCE EXACTLY (within 0.05pt rounding).")


if __name__ == "__main__":
    main()
