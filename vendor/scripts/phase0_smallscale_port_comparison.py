"""
Step 3 of the 2026-08-10 pre-scaling checks (docs/V5_LOG.md step 37) -- THE DECISIVE TEST
before committing to full-scale generation + retrain. Generates a small corrected-format
training set (combined port: corrected_blend_timing + windowed_examples +
content_majority_labeling) and a matched-size baseline set (current generate_dataset()
format), trains two STGT models from IDENTICAL initial weights (torch.manual_seed reset
immediately before each model's construction, before any other torch RNG consumption),
differing only in which dataset they saw, then evaluates both on the same
eval_trajectories.py chain-2 population (seed=999, the standing ceiling battery's own seed)
for pair and threat accuracy.

Target sizes matched by FINAL EXAMPLE COUNT, not hops requested: n_per_formation=300 steady-
state each (identical for both, unaffected by the port flags), ~900 transitioning examples
each (baseline: n_transition=900 directly; corrected: n_transition sized up by step 36's
2.06x compensation factor to compensate for the ~51% exclusion rate).

Usage (run inside tmux -- trains 2 models, each a few minutes on a single GPU):
    python scripts/phase0_smallscale_port_comparison.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.config import Config  # noqa: E402
from swarm_intent.data import generate_dataset, save_splits, split_and_normalize  # noqa: E402
from swarm_intent.train import train, get_device  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

N_PER_FORMATION = 300
TARGET_TRANSITIONING = 900
COMPENSATION_FACTOR = 2.06        # docs/V5_LOG.md step 36 -- applies to WINDOWS needed, not hops
WINDOWS_PER_HOP = 6.12            # docs/V5_LOG.md step 36 (mean across 5 seeds)
# 2026-08-10 bug fix: an earlier version of this script computed
# N_TRANSITION_CORRECTED = TARGET_TRANSITIONING * COMPENSATION_FACTOR directly, treating the
# compensation factor as "hops needed" -- it isn't; it's "windows needed", and each hop
# already yields ~6.12 windows BEFORE the exclusion filter. That bug requested 1854 hops,
# which produced 5656 kept transitioning examples against a target of 900 (a 7756-example,
# 73%-transitioning dataset vs baseline's 3000-example, 30%-transitioning one) -- a severe,
# confounding class-imbalance difference between the two datasets that had nothing to do with
# the port design itself. Corrected: hops = target_kept / (windows_per_hop * keep_rate).
N_TRANSITION_CORRECTED = round(TARGET_TRANSITIONING / (WINDOWS_PER_HOP / COMPENSATION_FACTOR))

DATA_SEED = 7  # same seed used throughout steps 34-36, for continuity
MODEL_INIT_SEED = 4242  # arbitrary fixed seed, shared by BOTH models -- identical initial weights
EPOCHS = 60
EVAL_N = 500
EVAL_SEED = 999  # the standing ceiling-battery seed, unchanged, per instruction ("the
                 # already-validated, fixed ruler")

DATA_DIR_BASELINE = REPO / "swarm_data_smallscale_baseline"
DATA_DIR_CORRECTED = REPO / "swarm_data_smallscale_corrected"


def build_and_save(data_dir: Path, X, y, names, cfg_seed):
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    cfg = Config(seed=cfg_seed, n_classes=8, data_dir=str(data_dir))
    splits = split_and_normalize(X, y, cfg)
    save_splits(splits, cfg)  # writes X/y *.npy + combined norm_stats.npy
    # save_splits does NOT write separate train_mean.npy/train_std.npy (only the combined
    # norm_stats.npy) -- but sliding_window_inference (used at eval time below) expects
    # them as separate files, matching every other eval script's convention. Write them here
    # rather than relying on stale files in a different data dir.
    np.save(data_dir / "train_mean.npy", np.array(splits["train_mean"], dtype=np.float32))
    np.save(data_dir / "train_std.npy", np.array(splits["train_std"], dtype=np.float32))
    with open(data_dir / "class_names.json", "w") as f:
        json.dump(names, f)
    return cfg


def main():
    print("=" * 100)
    print("PHASE A: generate both datasets")
    print("=" * 100)

    rng_cfg = Config(seed=DATA_SEED)
    print(f"baseline: n_per_formation={N_PER_FORMATION}, n_transition={TARGET_TRANSITIONING} (direct, old format)")
    X_base, y_base, names_base = generate_dataset(
        rng_cfg, n_per_formation=N_PER_FORMATION, n_timesteps=50, include_transitions=True,
        n_transition=TARGET_TRANSITIONING,
    )
    print(f"baseline dataset: X={X_base.shape}, transitioning examples={TARGET_TRANSITIONING} exactly (no exclusion in old format)")

    rng_cfg2 = Config(seed=DATA_SEED)
    print(f"corrected: n_per_formation={N_PER_FORMATION}, n_transition={N_TRANSITION_CORRECTED} hops requested (combined port)")
    X_corr, y_corr, names_corr, diag = generate_dataset(
        rng_cfg2, n_per_formation=N_PER_FORMATION, n_timesteps=50, include_transitions=True,
        n_transition=N_TRANSITION_CORRECTED, corrected_blend_timing=True, windowed_examples=True,
        content_majority_labeling=True, return_diagnostics=True,
    )
    n_kept_transitioning = diag["n_examples_kept"]
    print(f"corrected dataset: X={X_corr.shape}, transitioning examples kept={n_kept_transitioning} "
         f"(target was {TARGET_TRANSITIONING}, excluded={diag['n_excluded']})")
    print(f"actual keep rate this run: {n_kept_transitioning/(n_kept_transitioning+diag['n_excluded']):.1%}")

    cfg_base = build_and_save(DATA_DIR_BASELINE, X_base, y_base, names_base, cfg_seed=DATA_SEED)
    cfg_corr = build_and_save(DATA_DIR_CORRECTED, X_corr, y_corr, names_corr, cfg_seed=DATA_SEED)

    print("\n" + "=" * 100)
    print("PHASE B: train two models from IDENTICAL initial weights")
    print("=" * 100)
    device = get_device()
    print(f"device={device}")

    print(f"\n--- training BASELINE model (seed reset to {MODEL_INIT_SEED} immediately before construction) ---")
    torch.manual_seed(MODEL_INIT_SEED)
    train_cfg_base = Config(seed=DATA_SEED, n_classes=8, data_dir=str(DATA_DIR_BASELINE), epochs=EPOCHS)
    model_base, hist_base, test_acc_base = train(train_cfg_base, device=device, ckpt_name="best_model_smallscale_baseline.pt")
    print(f"baseline: test_acc={test_acc_base:.4f}")

    print(f"\n--- training CORRECTED model (seed reset to {MODEL_INIT_SEED} immediately before construction) ---")
    torch.manual_seed(MODEL_INIT_SEED)
    train_cfg_corr = Config(seed=DATA_SEED, n_classes=8, data_dir=str(DATA_DIR_CORRECTED), epochs=EPOCHS)
    model_corr, hist_corr, test_acc_corr = train(train_cfg_corr, device=device, ckpt_name="best_model_smallscale_corrected.pt")
    print(f"corrected: test_acc={test_acc_corr:.4f}")

    print("\n" + "=" * 100)
    print("PHASE C: evaluate both on eval_trajectories.py chain-2 ceiling (seed=999)")
    print("=" * 100)

    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference
    from swarm_intent.coverage import classify_observation, BUCKET_A
    from swarm_intent.eval_trajectories import sample_chain, build_long_sequence_labeled, ground_truth_pair
    sys.path.insert(0, str(REPO / "llm_finetuning"))
    from build_sft_dataset import RULES  # noqa: E402

    def eval_checkpoint(ckpt_path, data_dir, label):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = STGTModel(ckpt["cfg"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        train_mean = np.load(data_dir / "train_mean.npy")
        train_std = np.load(data_dir / "train_std.npy")
        reg_mean = ckpt["reg_mean"]
        reg_std = ckpt["reg_std"]

        rng = np.random.default_rng(EVAL_SEED)
        reporter = Reporter(f"smallscale_eval_{label}", EVAL_N, rate_hint=8.0)
        n_chain2 = n_pair_correct = n_threat_correct = 0
        for i in range(EVAL_N):
            chain = sample_chain(rng)
            spread = float(rng.uniform(0.6, 1.8))
            noise_std = float(rng.uniform(0.15, 1.4))
            long_seq, true_labels = build_long_sequence_labeled(chain, rng, spread, noise_std)
            if len(chain) == 2:
                predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                                       train_mean, train_std, window_size=50, stride=10, dt=0.5)
                gt_pair = ground_truth_pair(chain)
                info = classify_observation(predictions, robust=False)
                n_chain2 += 1
                if info["bucket"] == BUCKET_A:
                    rec_pair = tuple(info["rules_key"])
                    if rec_pair == gt_pair:
                        n_pair_correct += 1
                    if rec_pair in RULES and gt_pair in RULES and RULES[rec_pair][0] == RULES[gt_pair][0]:
                        n_threat_correct += 1
            reporter.update(1, item=f"seq {i}")
        reporter.status = "done"
        reporter._write()
        return {"n_chain2": n_chain2, "pair_accuracy": n_pair_correct / n_chain2,
               "threat_accuracy": n_threat_correct / n_chain2}

    result_base = eval_checkpoint(DATA_DIR_BASELINE / "best_model_smallscale_baseline.pt", DATA_DIR_BASELINE, "baseline")
    result_corr = eval_checkpoint(DATA_DIR_CORRECTED / "best_model_smallscale_corrected.pt", DATA_DIR_CORRECTED, "corrected")

    print("\n" + "=" * 100)
    print("RESULT: small-scale before/after ceiling comparison (chain-2 only, seed=999, n=%d sampled)" % EVAL_N)
    print("=" * 100)
    print(f"BASELINE  (old format, n_transition={TARGET_TRANSITIONING}):     "
         f"n_chain2={result_base['n_chain2']}  pair_acc={result_base['pair_accuracy']:.1%}  "
         f"threat_acc={result_base['threat_accuracy']:.1%}  test_acc={test_acc_base:.4f}")
    print(f"CORRECTED (combined port, {n_kept_transitioning} kept): "
         f"n_chain2={result_corr['n_chain2']}  pair_acc={result_corr['pair_accuracy']:.1%}  "
         f"threat_acc={result_corr['threat_accuracy']:.1%}  test_acc={test_acc_corr:.4f}")
    delta_pair = result_corr['pair_accuracy'] - result_base['pair_accuracy']
    delta_threat = result_corr['threat_accuracy'] - result_base['threat_accuracy']
    print(f"\nDELTA: pair_acc {delta_pair:+.1%}   threat_acc {delta_threat:+.1%}")
    print("CORRECTED WINS" if delta_pair > 0 and delta_threat > 0 else
         "MIXED/NO CLEAR WIN" if (delta_pair > 0) != (delta_threat > 0) else "BASELINE WINS OR TIE")

    out = {
        "n_per_formation": N_PER_FORMATION, "target_transitioning": TARGET_TRANSITIONING,
        "n_transition_corrected_requested": N_TRANSITION_CORRECTED,
        "n_kept_transitioning_corrected": n_kept_transitioning,
        "epochs": EPOCHS, "model_init_seed": MODEL_INIT_SEED, "eval_seed": EVAL_SEED, "eval_n": EVAL_N,
        "baseline": {**result_base, "test_acc": test_acc_base},
        "corrected": {**result_corr, "test_acc": test_acc_corr},
        "delta_pair_accuracy": delta_pair, "delta_threat_accuracy": delta_threat,
    }
    with open(REPO / "evaluation" / "phase0_smallscale_port_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved evaluation/phase0_smallscale_port_comparison.json")


if __name__ == "__main__":
    main()
