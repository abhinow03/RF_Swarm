"""
The B-vs-C decisive test (docs/V5_LOG.md step 48): re-runs the exact 5-seed comparison from
steps 42-44/46-47, but with n_transition resized from 303 (hops, tuned to match baseline's
KEPT-example count) to 900 (hops, tuned to match baseline's INDEPENDENT-sample count exactly
-- 900 hops = 900 independent draws, identical to baseline's 900). Verified before this script
was written: n_hops_sampled=900 exactly, x5 seeds=4500, matching baseline's 4500 total.

Accepted side effect, not a bug: kept transitioning examples now ~2753/seed (vs baseline's
900) since windowing yields ~6.12 examples/hop even after the ~51% exclusion filter --
corrected's dataset is now larger and class-imbalanced toward transitioning relative to
baseline. This run tests whether matching INDEPENDENT sample count (not kept-example count
or class balance) closes the threat_acc gap -- if it does, verdict B is confirmed and the fix
is exactly this resize; if not, verdict C stands.

Same seeds (20-24), same eval_trajectories.py protocol (seed=999 ruler, EVAL_N=500), same
Welch's t-test format as steps 42-44.

Usage (run inside tmux -- 10 training runs, corrected side now larger, ~40-60 min):
    python scripts/phase0_variance_matched_independent.py
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
N_TRANSITION_CORRECTED = 900  # CHANGED from 303 -- matches baseline's independent-sample count directly
EPOCHS = 60
EVAL_N = 500
EVAL_SEED = 999

SEEDS = [20, 21, 22, 23, 24]  # identical to steps 42-44/46-47

DATA_DIR_BASELINE = REPO / "swarm_data_matchindep_baseline"
DATA_DIR_CORRECTED = REPO / "swarm_data_matchindep_corrected"


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
    reporter = Reporter(f"matchindep_eval_{label}", EVAL_N, rate_hint=8.0)
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
    Xc, yc, nc, diag = generate_dataset(cfg_c, n_per_formation=N_PER_FORMATION, n_timesteps=50,
                                        include_transitions=True, n_transition=N_TRANSITION_CORRECTED,
                                        corrected_blend_timing=True, windowed_examples=True,
                                        content_majority_labeling=True, return_diagnostics=True)
    build_and_save(DATA_DIR_CORRECTED, Xc, yc, nc, cfg_seed=seed, robust=True)
    print(f"seed {seed}: baseline X={Xb.shape}; corrected X={Xc.shape} "
         f"(hops={diag['n_hops_sampled']}, kept={diag['n_examples_kept']})")

    torch.manual_seed(seed)
    train_cfg_b = Config(seed=seed, n_classes=8, data_dir=str(DATA_DIR_BASELINE), epochs=EPOCHS)
    _, _, test_acc_b = train(train_cfg_b, device=device, ckpt_name="best_model.pt")

    torch.manual_seed(seed)
    train_cfg_c = Config(seed=seed, n_classes=8, data_dir=str(DATA_DIR_CORRECTED), epochs=EPOCHS)
    _, _, test_acc_c = train(train_cfg_c, device=device, ckpt_name="best_model.pt")

    result_b = eval_checkpoint(DATA_DIR_BASELINE / "best_model.pt", DATA_DIR_BASELINE, device, f"base_s{seed}")
    result_c = eval_checkpoint(DATA_DIR_CORRECTED / "best_model.pt", DATA_DIR_CORRECTED, device, f"corr_s{seed}")

    row = {"seed": seed, "baseline": {"test_acc": test_acc_b, **result_b},
          "corrected": {"test_acc": test_acc_c, **result_c},
          "corrected_n_hops": diag["n_hops_sampled"], "corrected_n_kept": diag["n_examples_kept"]}
    print(f"seed {seed} RESULT: baseline test_acc={test_acc_b:.4f} pair={result_b['pair_accuracy']:.1%} "
         f"threat={result_b['threat_accuracy']:.1%}  |  corrected test_acc={test_acc_c:.4f} "
         f"pair={result_c['pair_accuracy']:.1%} threat={result_c['threat_accuracy']:.1%}")
    return row


def main():
    device = get_device()
    print(f"device={device}, seeds={SEEDS}, N_TRANSITION_CORRECTED={N_TRANSITION_CORRECTED}")
    rows = []
    for seed in SEEDS:
        rows.append(run_one(seed, device))
        with open(REPO / "evaluation" / "phase0_variance_matched_independent.json", "w") as f:
            json.dump(rows, f, indent=2)

    print("\n" + "=" * 100)
    print("FULL SUMMARY")
    print("=" * 100)
    for metric_key, label in [("test_acc", "test_acc"), ("threat_accuracy", "chain2_threat_acc"),
                              ("pair_accuracy", "chain2_pair_acc")]:
        b_vals = np.array([r["baseline"][metric_key] for r in rows])
        c_vals = np.array([r["corrected"][metric_key] for r in rows])
        print(f"\n{label}:")
        print(f"  baseline : {[f'{v:.4f}' for v in b_vals]}  mean={b_vals.mean():.4f} std={b_vals.std():.4f}")
        print(f"  corrected: {[f'{v:.4f}' for v in c_vals]}  mean={c_vals.mean():.4f} std={c_vals.std():.4f}")
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(c_vals, b_vals, equal_var=False)
        print(f"  Welch's t-test (corrected vs baseline): t={t_stat:.3f} p={p_val:.4f}")

    with open(REPO / "evaluation" / "phase0_variance_matched_independent.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nsaved evaluation/phase0_variance_matched_independent.json")


if __name__ == "__main__":
    main()
