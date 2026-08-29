"""
AUDIT.md sec AM follow-up, step 2: isolate the dwell-time/symmetrization generator fix's real
effect from population resampling. Step 3's diff already proved OLD/MID/CURRENT are
genuinely different random draws (98.8-99.2% divergent chains) -- so the historically-cited
"58.7%->83.0%/77.3%" and "39.9%->65.8%" deltas conflate the fix's mechanistic effect with an
ordinary large-n resample. This script builds a PAIRED design instead: for each of the same
1000 trajectory indices, draw (chain, spread, noise_std) ONCE from a trajectory-local RNG
(np.random.default_rng(SeedSequence([999, i]))), fork the RNG state right after those shared
draws, then run each of the 3 known hop-timing formulas (OLD / MID=dwell-fix-only /
CURRENT=dwell-fix+symmetrized) from that SAME forked state. All 3 variants therefore share the
identical 1000-chain population (same formation sequence, same spread, same noise_std per
index) -- only the transition-timing formula differs. Scored with the CURRENT, already-fixed
guard logic and the single locked checkpoint throughout, so the ONLY thing that can move the
number between variants is the timing formula itself.

No training, no corpus changes -- CPU inference only.

Usage:
    python scripts/rule0_am_isolate_fixes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))
sys.path.insert(0, str(REPO / "scripts"))

from swarm_intent.config import BASE_FORMATIONS, TRANSITION_CLASS  # noqa: E402
from swarm_intent.coverage import classify_observation, BUCKET_A  # noqa: E402
from swarm_intent.eval_trajectories import ground_truth_pair  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402
from rule0_2b_regenerate_populations import (  # noqa: E402
    sample_chain_OLD, build_long_sequence_labeled_OLD,
    sample_chain_MID, build_long_sequence_labeled_MID,
    build_long_sequence_labeled_CURRENT,
)

SEED_BASE = 999
N = 1000
DATA_DIR = REPO / "swarm_data"
CHECKPOINT = DATA_DIR / "best_model.pt"

FORMULAS = {
    "OLD_pre_dwell_fix": build_long_sequence_labeled_OLD,
    "MID_dwell_fix_only": build_long_sequence_labeled_MID,
    "CURRENT_dwell_fix_and_symmetrize": build_long_sequence_labeled_CURRENT,
}

# Historically-cited, NON-isolated (population-confounded) figures, for side-by-side reporting.
HISTORICAL_COMBINED = {
    "pooled_threat": {"OLD_pre_dwell_fix": 0.587, "CURRENT_dwell_fix_and_symmetrize": 0.830,
                      "note": "58.7% (step11 trimfix, OLD population) -> 83.0% (CEILING.md 'current state', CURRENT population)"},
    "chain_2_pair": {"OLD_pre_dwell_fix": 0.187, "MID_dwell_fix_only": 0.399,
                     "CURRENT_dwell_fix_and_symmetrize": 0.658,
                     "note": "18.7% (baseline) -> 39.9% (step25 dwell-fix, MID-equivalent population) -> 65.8% (step26c symmetrized, CURRENT population)"},
}


def build_paired_trajectory(i):
    """Returns {formula_name: (chain, long_seq, true_labels)} -- all 3 share the identical
    chain/spread/noise_std, forked from the same RNG state."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, i]))
    chain = [str(f) for f in _sample_chain_shared(rng)]
    spread = float(rng.uniform(0.6, 1.8))
    noise_std = float(rng.uniform(0.15, 1.4))
    fork_state = rng.bit_generator.state

    out = {}
    for name, build_fn in FORMULAS.items():
        forked_rng = np.random.default_rng()
        forked_rng.bit_generator.state = fork_state
        long_seq, true_labels = build_fn(chain, forked_rng, spread, noise_std)
        out[name] = (chain, long_seq, true_labels)
    return chain, out


def _sample_chain_shared(rng):
    # sample_chain is byte-identical across OLD/MID/CURRENT (verified in rule0_2b_step1_inventory
    # -- only build_long_sequence_labeled ever diverged) -- use one canonical copy.
    return sample_chain_OLD(rng)


def main():
    import torch
    import swarm_intent.stgt.model as stgt_model_module
    device = torch.device("cpu")
    stgt_model_module.device = device
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = STGTModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    # pass 1 (cheap, no model): determine chain identity per index, filter to pair-eligible
    pair_eligible = []
    for i in range(N):
        rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, i]))
        chain = [str(f) for f in _sample_chain_shared(rng)]
        gt = ground_truth_pair(chain)
        if gt is not None:
            pair_eligible.append(i)
    print(f"pair-eligible indices (shared across all 3 formulas): {len(pair_eligible)}/{N}")

    per_record = {name: [] for name in FORMULAS}
    for count, i in enumerate(pair_eligible):
        chain, variants = build_paired_trajectory(i)
        gt_pair = ground_truth_pair(chain)
        for name, (chain_v, long_seq, true_labels) in variants.items():
            predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                                   train_mean, train_std, window_size=50, stride=10, dt=0.5)
            bucket_t = classify_observation(predictions, robust=True)
            rec_pair = tuple(bucket_t["rules_key"]) if bucket_t["bucket"] == BUCKET_A else None
            per_record[name].append({
                "i": i, "chain_length": len(chain), "gt_pair": list(gt_pair),
                "recovered_pair": list(rec_pair) if rec_pair else None,
            })
        if (count + 1) % 50 == 0:
            print(f"  scored {count + 1}/{len(pair_eligible)} paired trajectories...")

    def score(recs):
        n = len(recs)
        pair_correct = threat_correct = 0
        for r in recs:
            gt = tuple(r["gt_pair"])
            true_threat = RULES[gt][0]
            rec_pair = r["recovered_pair"]
            if rec_pair is None:
                continue
            rec_pair = tuple(rec_pair)
            if rec_pair == gt:
                pair_correct += 1
            if RULES[rec_pair][0] == true_threat:
                threat_correct += 1
        return {"n": n, "pair_accuracy": pair_correct / n if n else None,
               "threat_accuracy": threat_correct / n if n else None}

    results = {}
    for name, recs in per_record.items():
        strata = {
            "pooled": recs,
            "chain_1": [r for r in recs if r["chain_length"] == 1],
            "chain_2": [r for r in recs if r["chain_length"] == 2],
        }
        results[name] = {s: score(recs_s) for s, recs_s in strata.items()}

    print("\n" + "=" * 100)
    print("ISOLATED FIX EFFECT -- same 1000-chain paired population, only the timing formula varies")
    print("=" * 100)
    for stratum in ("pooled", "chain_1", "chain_2"):
        print(f"\n--- {stratum} ---")
        for name in FORMULAS:
            r = results[name][stratum]
            print(f"  {name}: n={r['n']}  threat={r['threat_accuracy']:.1%}  pair={r['pair_accuracy']:.1%}"
                 if r['n'] else f"  {name}: n=0")

    print("\n" + "=" * 100)
    print("OLD COMBINED (population-confounded, historically cited) vs NEW ISOLATED (this run) -- side by side")
    print("=" * 100)
    pooled_threat_isolated = {name: results[name]["pooled"]["threat_accuracy"] for name in FORMULAS}
    chain2_pair_isolated = {name: results[name]["chain_2"]["pair_accuracy"] for name in FORMULAS}
    print("pooled threat accuracy:")
    print(f"  historical (population-confounded): OLD 58.7% -> CURRENT 83.0% (delta +24.3pt)")
    print(f"  isolated (same population):         OLD {pooled_threat_isolated['OLD_pre_dwell_fix']:.1%} -> "
         f"CURRENT {pooled_threat_isolated['CURRENT_dwell_fix_and_symmetrize']:.1%} "
         f"(delta {100*(pooled_threat_isolated['CURRENT_dwell_fix_and_symmetrize']-pooled_threat_isolated['OLD_pre_dwell_fix']):+.1f}pt)")
    print("chain-2 pair accuracy:")
    print(f"  historical (population-confounded): OLD 18.7% -> MID(dwell-fix) 39.9% -> CURRENT(symmetrized) 65.8%")
    print(f"  isolated (same population):         OLD {chain2_pair_isolated['OLD_pre_dwell_fix']:.1%} -> "
         f"MID {chain2_pair_isolated['MID_dwell_fix_only']:.1%} -> "
         f"CURRENT {chain2_pair_isolated['CURRENT_dwell_fix_and_symmetrize']:.1%}")

    out = {
        "seed_base": SEED_BASE, "n_pair_eligible": len(pair_eligible),
        "results_by_formula": results,
        "historical_combined_figures": HISTORICAL_COMBINED,
        "isolated_pooled_threat": pooled_threat_isolated,
        "isolated_chain2_pair": chain2_pair_isolated,
        "method": "paired design -- same chain/spread/noise_std per trajectory index across all "
                 "3 formulas, forked from a shared per-trajectory RNG state right after the "
                 "shared draws, so only the hop-timing formula differs between variants.",
    }
    out_path = REPO / "evaluation" / "rule0_am_isolated_fix_effects.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
