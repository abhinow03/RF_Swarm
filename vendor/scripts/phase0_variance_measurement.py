"""
Steps 1-2 of the 2026-08-10 variance-characterization session (docs/V5_LOG.md step 42):
measures baseline and corrected-port run-to-run variance at 5 seeds each, same small-scale
protocol as steps 37/39/41 (n_per_formation=300, baseline n_transition=900, corrected
n_transition=303 with robust=True normalization and acceleration uncapped -- commit f6c5ffb's
config), before trusting any further single-run comparison between them.

Each seed varies BOTH the data-generation seed and the model-init seed together (so this
captures genuine data/init variance, not just GPU/cuDNN non-determinism alone) -- though
steps 37/39/41 already showed real run-to-run variance existed even at IDENTICAL nominal
seeds (data seed=7, init seed=4242, unchanged across all three), meaning GPU training
non-determinism alone is already a real contributor this measurement will also capture.

The eval ruler stays FIXED: eval_trajectories.py's own seed=999 population is unchanged
across every run -- only the TRAINING seed varies, matching "same test, different training
runs" exactly.

Usage (run inside tmux -- 10 training runs, ~30-45 min total):
    python scripts/phase0_variance_measurement.py
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
N_TRANSITION_CORRECTED = 303  # same as steps 37/39/41
EPOCHS = 60
EVAL_N = 500
EVAL_SEED = 999  # FIXED ruler, unchanged across every run

SEEDS = [20, 21, 22, 23, 24]  # fresh, disjoint from every seed used elsewhere in this program

DATA_DIR_BASELINE = REPO / "swarm_data_variance_baseline"
DATA_DIR_CORRECTED = REPO / "swarm_data_variance_corrected"


def build_and_save(data_dir, X, y, names, cfg_seed, robust):
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


def eval_checkpoint(ckpt_path, data_dir, device, label):
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference
    from swarm_intent.coverage import classify_observation, BUCKET_A
    from swarm_intent.eval_trajectories import sample_chain, build_long_sequence_labeled, ground_truth_pair
    sys.path.insert(0, str(REPO / "llm_finetuning"))
    from build_sft_dataset import RULES

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = STGTModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    train_mean = np.load(data_dir / "train_mean.npy")
    train_std = np.load(data_dir / "train_std.npy")
    reg_mean = ckpt["reg_mean"]
    reg_std = ckpt["reg_std"]

    rng = np.random.default_rng(EVAL_SEED)
    reporter = Reporter(f"variance_eval_{label}", EVAL_N, rate_hint=8.0)
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


def run_one(seed, device):
    print(f"\n{'='*100}\nSEED {seed}\n{'='*100}")

    cfg_b = Config(seed=seed)
    Xb, yb, nb = generate_dataset(cfg_b, n_per_formation=N_PER_FORMATION, n_timesteps=50,
                                  include_transitions=True, n_transition=TARGET_TRANSITIONING)
    build_and_save(DATA_DIR_BASELINE, Xb, yb, nb, cfg_seed=seed, robust=False)

    cfg_c = Config(seed=seed)
    Xc, yc, nc = generate_dataset(cfg_c, n_per_formation=N_PER_FORMATION, n_timesteps=50,
                                  include_transitions=True, n_transition=N_TRANSITION_CORRECTED,
                                  corrected_blend_timing=True, windowed_examples=True,
                                  content_majority_labeling=True)
    build_and_save(DATA_DIR_CORRECTED, Xc, yc, nc, cfg_seed=seed, robust=True)

    print(f"seed {seed}: baseline X={Xb.shape}, corrected X={Xc.shape}")

    torch.manual_seed(seed)
    train_cfg_b = Config(seed=seed, n_classes=8, data_dir=str(DATA_DIR_BASELINE), epochs=EPOCHS)
    _, _, test_acc_b = train(train_cfg_b, device=device, ckpt_name="best_model.pt")

    torch.manual_seed(seed)
    train_cfg_c = Config(seed=seed, n_classes=8, data_dir=str(DATA_DIR_CORRECTED), epochs=EPOCHS)
    _, _, test_acc_c = train(train_cfg_c, device=device, ckpt_name="best_model.pt")

    result_b = eval_checkpoint(DATA_DIR_BASELINE / "best_model.pt", DATA_DIR_BASELINE, device, f"base_s{seed}")
    result_c = eval_checkpoint(DATA_DIR_CORRECTED / "best_model.pt", DATA_DIR_CORRECTED, device, f"corr_s{seed}")

    row = {
        "seed": seed,
        "baseline": {"test_acc": test_acc_b, **result_b},
        "corrected": {"test_acc": test_acc_c, **result_c},
    }
    print(f"seed {seed} RESULT: baseline test_acc={test_acc_b:.4f} pair={result_b['pair_accuracy']:.1%} "
         f"threat={result_b['threat_accuracy']:.1%}  |  corrected test_acc={test_acc_c:.4f} "
         f"pair={result_c['pair_accuracy']:.1%} threat={result_c['threat_accuracy']:.1%}")
    return row


def main():
    device = get_device()
    print(f"device={device}, seeds={SEEDS}")
    rows = []
    for seed in SEEDS:
        rows.append(run_one(seed, device))
        with open(REPO / "evaluation" / "phase0_variance_measurement.json", "w") as f:
            json.dump(rows, f, indent=2)

    print("\n" + "=" * 100)
    print("FULL SUMMARY")
    print("=" * 100)
    for metric_key, label in [("test_acc", "test_acc"), ("pair_accuracy", "chain2_pair_acc"),
                              ("threat_accuracy", "chain2_threat_acc")]:
        b_vals = np.array([r["baseline"][metric_key] for r in rows])
        c_vals = np.array([r["corrected"][metric_key] for r in rows])
        print(f"\n{label}:")
        print(f"  baseline : {[f'{v:.4f}' for v in b_vals]}  mean={b_vals.mean():.4f} std={b_vals.std():.4f} "
             f"min={b_vals.min():.4f} max={b_vals.max():.4f}")
        print(f"  corrected: {[f'{v:.4f}' for v in c_vals]}  mean={c_vals.mean():.4f} std={c_vals.std():.4f} "
             f"min={c_vals.min():.4f} max={c_vals.max():.4f}")
        try:
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(c_vals, b_vals, equal_var=False)
            print(f"  Welch's t-test (corrected vs baseline): t={t_stat:.3f} p={p_val:.4f}")
        except ImportError:
            print("  (scipy not available for t-test, reporting raw distributions only)")

    with open(REPO / "evaluation" / "phase0_variance_measurement.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nsaved evaluation/phase0_variance_measurement.json")


if __name__ == "__main__":
    main()
