"""
Step 1 of the 2026-08-10 mechanism-isolation follow-up (docs/V5_LOG.md step 41): tests
whether ROBUST (percentile-trimmed) normalization alone -- with acceleration UNCAPPED --
resolves the small-scale training collapse (docs/V5_LOG.md steps 37/39), to determine whether
the acceleration cap (step 39) was ever a real contributor or whether normalization distortion
(step 40) explains the whole thing on its own.

Identical protocol to scripts/phase0_smallscale_port_comparison.py (same seed=7, same
n_per_formation=300/n_transition=900(baseline)/303(corrected) sizing, same model_init_seed,
same eval_trajectories.py chain-2 comparison) with two changes:
  - src.swarm_intent.data.ACCEL_SPEED_CAP is monkeypatched to +inf for this run -- baseline
    is UNAFFECTED (it never calls generate_transition_sequence above 50 timesteps, so the cap
    never engages regardless of its value; proven in step 39).
  - The CORRECTED dataset's split_and_normalize call uses robust=True (percentile-trimmed
    global scalar, docs/V5_LOG.md step 41). Baseline keeps robust=False (its own scale is
    already homogeneous; changing it would be an unrelated, unneeded variable in this test).

Usage (run inside tmux):
    python scripts/phase0_smallscale_normalization_isolation.py
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

import swarm_intent.data as data_module  # noqa: E402
data_module.ACCEL_SPEED_CAP = float("inf")  # uncap -- must happen BEFORE generate_dataset call

from swarm_intent.config import Config  # noqa: E402
from swarm_intent.data import generate_dataset, save_splits, split_and_normalize  # noqa: E402
from swarm_intent.train import train, get_device  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

N_PER_FORMATION = 300
TARGET_TRANSITIONING = 900
COMPENSATION_FACTOR = 2.06
WINDOWS_PER_HOP = 6.12
N_TRANSITION_CORRECTED = round(TARGET_TRANSITIONING / (WINDOWS_PER_HOP / COMPENSATION_FACTOR))

DATA_SEED = 7
MODEL_INIT_SEED = 4242
EPOCHS = 60
EVAL_N = 500
EVAL_SEED = 999

DATA_DIR_BASELINE = REPO / "swarm_data_smallscale_baseline"
DATA_DIR_CORRECTED = REPO / "swarm_data_smallscale_corrected_uncapped_robust"


def build_and_save(data_dir: Path, X, y, names, cfg_seed, robust):
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    cfg = Config(seed=cfg_seed, n_classes=8, data_dir=str(data_dir))
    splits = split_and_normalize(X, y, cfg, robust=robust)
    save_splits(splits, cfg)
    np.save(data_dir / "train_mean.npy", np.array(splits["train_mean"], dtype=np.float32))
    np.save(data_dir / "train_std.npy", np.array(splits["train_std"], dtype=np.float32))
    with open(data_dir / "class_names.json", "w") as f:
        json.dump(names, f)
    return cfg, splits


def main():
    print(f"ACCEL_SPEED_CAP (monkeypatched) = {data_module.ACCEL_SPEED_CAP}")
    print("=" * 100)
    print("PHASE A: generate both datasets (baseline unaffected; corrected uncapped+robust)")
    print("=" * 100)

    rng_cfg = Config(seed=DATA_SEED)
    X_base, y_base, names_base = generate_dataset(
        rng_cfg, n_per_formation=N_PER_FORMATION, n_timesteps=50, include_transitions=True,
        n_transition=TARGET_TRANSITIONING,
    )
    print(f"baseline: X={X_base.shape}")

    rng_cfg2 = Config(seed=DATA_SEED)
    X_corr, y_corr, names_corr, diag = generate_dataset(
        rng_cfg2, n_per_formation=N_PER_FORMATION, n_timesteps=50, include_transitions=True,
        n_transition=N_TRANSITION_CORRECTED, corrected_blend_timing=True, windowed_examples=True,
        content_majority_labeling=True, return_diagnostics=True,
    )
    print(f"corrected (uncapped accel): X={X_corr.shape}, kept={diag['n_examples_kept']}, "
         f"excluded={diag['n_excluded']}")

    cfg_base, splits_base = build_and_save(DATA_DIR_BASELINE, X_base, y_base, names_base,
                                           cfg_seed=DATA_SEED, robust=False)
    cfg_corr, splits_corr = build_and_save(DATA_DIR_CORRECTED, X_corr, y_corr, names_corr,
                                           cfg_seed=DATA_SEED, robust=True)
    print(f"baseline  train_mean={splits_base['train_mean']:.3f} train_std={splits_base['train_std']:.3f}")
    print(f"corrected train_mean={splits_corr['train_mean']:.3f} train_std={splits_corr['train_std']:.3f} (robust=True)")

    print("\n" + "=" * 100)
    print("PHASE B: train two models from IDENTICAL initial weights")
    print("=" * 100)
    device = get_device()

    print(f"\n--- training BASELINE model (seed reset to {MODEL_INIT_SEED}) ---")
    torch.manual_seed(MODEL_INIT_SEED)
    train_cfg_base = Config(seed=DATA_SEED, n_classes=8, data_dir=str(DATA_DIR_BASELINE), epochs=EPOCHS)
    _, _, test_acc_base = train(train_cfg_base, device=device, ckpt_name="best_model.pt")
    print(f"baseline: test_acc={test_acc_base:.4f}")

    print(f"\n--- training CORRECTED model (uncapped accel, robust norm; seed reset to {MODEL_INIT_SEED}) ---")
    torch.manual_seed(MODEL_INIT_SEED)
    train_cfg_corr = Config(seed=DATA_SEED, n_classes=8, data_dir=str(DATA_DIR_CORRECTED), epochs=EPOCHS)
    _, _, test_acc_corr = train(train_cfg_corr, device=device, ckpt_name="best_model.pt")
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
        reporter = Reporter(f"norm_isolation_eval_{label}", EVAL_N, rate_hint=8.0)
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

    result_base = eval_checkpoint(DATA_DIR_BASELINE / "best_model.pt", DATA_DIR_BASELINE, "baseline")
    result_corr = eval_checkpoint(DATA_DIR_CORRECTED / "best_model.pt", DATA_DIR_CORRECTED, "corrected_uncapped_robust")

    print("\n" + "=" * 100)
    print("RESULT: uncapped acceleration + robust normalization vs baseline")
    print("=" * 100)
    print(f"BASELINE:                          test_acc={test_acc_base:.4f}  "
         f"pair_acc={result_base['pair_accuracy']:.1%}  threat_acc={result_base['threat_accuracy']:.1%}")
    print(f"CORRECTED (uncapped + robust norm): test_acc={test_acc_corr:.4f}  "
         f"pair_acc={result_corr['pair_accuracy']:.1%}  threat_acc={result_corr['threat_accuracy']:.1%}")

    out = {
        "accel_speed_cap": "inf (uncapped)", "corrected_normalization": "robust=True (P5-P95 trimmed)",
        "baseline": {**result_base, "test_acc": test_acc_base},
        "corrected": {**result_corr, "test_acc": test_acc_corr},
    }
    with open(REPO / "evaluation" / "phase0_normalization_isolation_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved evaluation/phase0_normalization_isolation_result.json")


if __name__ == "__main__":
    main()
