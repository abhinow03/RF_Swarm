"""
Diagnosis follow-up to docs/V5_LOG.md steps 42-45: WHERE the corrected-port threat_acc
deficit (72.46%+-2.46 vs baseline 80.70%+-5.49, p=0.037, step 44) concentrates. Re-trains the
same 5-seed comparison (seeds 20-24, same protocol as steps 42-44) but this time:
  - saves checkpoints + per-seed data dirs persistently (not overwritten/deleted), so this
    doesn't have to be re-run again for a follow-up question
  - records per-(a,b)-pair AND per-formation pair/threat correctness, not just pooled
  - saves each run's training history (train/val loss+acc per epoch) for the training-
    dynamics check (one baseline seed vs one corrected seed, converged vs undertrained)

Threat-ceiling terms reported first throughout, per instruction (pair accuracy second) --
threat_acc is what the 60-65%/70% floor discussion is actually about, given pair-level and
threat-level ceilings have diverged (docs/CEILING.md).

Usage (run inside tmux -- 10 training runs, ~30-45 min):
    python scripts/phase0_variance_diagnosis.py
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
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
N_TRANSITION_CORRECTED = 303
EPOCHS = 60
EVAL_N = 500
EVAL_SEED = 999

SEEDS = [20, 21, 22, 23, 24]  # identical to steps 42-44, for direct comparability

CKPT_ROOT = REPO / "checkpoints_variance_diagnosis"


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


def eval_checkpoint_detailed(ckpt_path, data_dir, device, label):
    """Like the eval in phase0_variance_measurement.py, but records per-trajectory
    (formation_a, formation_b, pair_correct, threat_correct) for later breakdown."""
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
    reporter = Reporter(f"diag_eval_{label}", EVAL_N, rate_hint=8.0)
    records = []
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
            pair_correct = threat_correct = False
            if info["bucket"] == BUCKET_A:
                rec_pair = tuple(info["rules_key"])
                pair_correct = rec_pair == gt_pair
                if rec_pair in RULES and gt_pair in RULES:
                    threat_correct = RULES[rec_pair][0] == RULES[gt_pair][0]
            records.append({"a": gt_pair[0], "b": gt_pair[1],
                           "pair_correct": bool(pair_correct), "threat_correct": bool(threat_correct)})
        reporter.update(1, item=f"seq {i}")
    reporter.status = "done"
    reporter._write()
    return records


def main():
    device = get_device()
    print(f"device={device}, seeds={SEEDS}")
    CKPT_ROOT.mkdir(exist_ok=True)

    all_results = {"baseline": {}, "corrected": {}}
    histories = {"baseline": {}, "corrected": {}}

    for seed in SEEDS:
        print(f"\n{'='*100}\nSEED {seed}\n{'='*100}")
        dd_base = REPO / f"swarm_data_diag_baseline_s{seed}"
        dd_corr = REPO / f"swarm_data_diag_corrected_s{seed}"

        cfg_b = Config(seed=seed)
        Xb, yb, nb = generate_dataset(cfg_b, n_per_formation=N_PER_FORMATION, n_timesteps=50,
                                      include_transitions=True, n_transition=TARGET_TRANSITIONING)
        build_and_save(dd_base, Xb, yb, nb, cfg_seed=seed, robust=False)

        cfg_c = Config(seed=seed)
        Xc, yc, nc = generate_dataset(cfg_c, n_per_formation=N_PER_FORMATION, n_timesteps=50,
                                      include_transitions=True, n_transition=N_TRANSITION_CORRECTED,
                                      corrected_blend_timing=True, windowed_examples=True,
                                      content_majority_labeling=True)
        build_and_save(dd_corr, Xc, yc, nc, cfg_seed=seed, robust=True)

        torch.manual_seed(seed)
        train_cfg_b = Config(seed=seed, n_classes=8, data_dir=str(dd_base), epochs=EPOCHS)
        _, hist_b, test_acc_b = train(train_cfg_b, device=device, ckpt_name="best_model.pt")
        histories["baseline"][seed] = hist_b

        torch.manual_seed(seed)
        train_cfg_c = Config(seed=seed, n_classes=8, data_dir=str(dd_corr), epochs=EPOCHS)
        _, hist_c, test_acc_c = train(train_cfg_c, device=device, ckpt_name="best_model.pt")
        histories["corrected"][seed] = hist_c

        # persist checkpoints + norm stats to a stable location, keyed by seed
        for label, dd in (("baseline", dd_base), ("corrected", dd_corr)):
            dest = CKPT_ROOT / f"{label}_s{seed}"
            dest.mkdir(exist_ok=True)
            shutil.copy(dd / "best_model.pt", dest / "best_model.pt")
            shutil.copy(dd / "train_mean.npy", dest / "train_mean.npy")
            shutil.copy(dd / "train_std.npy", dest / "train_std.npy")

        records_b = eval_checkpoint_detailed(dd_base / "best_model.pt", dd_base, device, f"base_s{seed}")
        records_c = eval_checkpoint_detailed(dd_corr / "best_model.pt", dd_corr, device, f"corr_s{seed}")
        all_results["baseline"][seed] = {"test_acc": test_acc_b, "records": records_b}
        all_results["corrected"][seed] = {"test_acc": test_acc_c, "records": records_c}

        pair_acc_b = np.mean([r["pair_correct"] for r in records_b])
        threat_acc_b = np.mean([r["threat_correct"] for r in records_b])
        pair_acc_c = np.mean([r["pair_correct"] for r in records_c])
        threat_acc_c = np.mean([r["threat_correct"] for r in records_c])
        print(f"seed {seed}: baseline threat={threat_acc_b:.1%} pair={pair_acc_b:.1%}  |  "
             f"corrected threat={threat_acc_c:.1%} pair={pair_acc_c:.1%}")

        # incremental save
        with open(REPO / "evaluation" / "phase0_variance_diagnosis_raw.json", "w") as f:
            json.dump(all_results, f, indent=2)
        with open(REPO / "evaluation" / "phase0_variance_diagnosis_histories.json", "w") as f:
            json.dump(histories, f, indent=2)

    # ---- per-pair / per-formation breakdown, pooled across all 5 seeds ----
    print("\n" + "=" * 100)
    print("PER-(a,b)-PAIR BREAKDOWN, pooled across 5 seeds -- THREAT terms first")
    print("=" * 100)
    pair_stats = defaultdict(lambda: {"baseline": {"n": 0, "threat": 0, "pair": 0},
                                      "corrected": {"n": 0, "threat": 0, "pair": 0}})
    for label in ("baseline", "corrected"):
        for seed in SEEDS:
            for r in all_results[label][seed]["records"]:
                key = (r["a"], r["b"])
                pair_stats[key][label]["n"] += 1
                pair_stats[key][label]["threat"] += int(r["threat_correct"])
                pair_stats[key][label]["pair"] += int(r["pair_correct"])

    print(f"{'pair':<32} {'n_b':>5} {'threat_b':>9} {'pair_b':>7}   {'n_c':>5} {'threat_c':>9} {'pair_c':>7}   {'threat_gap':>10}")
    rows = []
    for pair, s in sorted(pair_stats.items()):
        b, c = s["baseline"], s["corrected"]
        tb = b["threat"] / b["n"] if b["n"] else float("nan")
        pb = b["pair"] / b["n"] if b["n"] else float("nan")
        tc = c["threat"] / c["n"] if c["n"] else float("nan")
        pc = c["pair"] / c["n"] if c["n"] else float("nan")
        gap = tc - tb if b["n"] and c["n"] else float("nan")
        rows.append((pair, b["n"], tb, pb, c["n"], tc, pc, gap))
        print(f"{str(pair):<32} {b['n']:>5} {tb:>8.1%} {pb:>6.1%}   {c['n']:>5} {tc:>8.1%} {pc:>6.1%}   {gap:>+9.1%}" if b["n"] and c["n"] else
             f"{str(pair):<32} {b['n']:>5} {'--':>8} {'--':>6}   {c['n']:>5} {'--':>8} {'--':>6}   {'--':>9}")

    gaps = np.array([r[7] for r in rows if not np.isnan(r[7])])
    print(f"\nthreat_acc gap by pair: mean={gaps.mean():+.1%} std={gaps.std():.1%} min={gaps.min():+.1%} max={gaps.max():+.1%}")

    print("\n" + "=" * 100)
    print("PER-FORMATION BREAKDOWN (appears as EITHER a or b), pooled -- THREAT terms first")
    print("=" * 100)
    formation_stats = defaultdict(lambda: {"baseline": {"n": 0, "threat": 0}, "corrected": {"n": 0, "threat": 0}})
    for label in ("baseline", "corrected"):
        for seed in SEEDS:
            for r in all_results[label][seed]["records"]:
                for f in (r["a"], r["b"]):
                    formation_stats[f][label]["n"] += 1
                    formation_stats[f][label]["threat"] += int(r["threat_correct"])

    print(f"{'formation':<15} {'n_b':>6} {'threat_b':>9}   {'n_c':>6} {'threat_c':>9}   {'gap':>8}")
    frows = []
    for formation, s in sorted(formation_stats.items()):
        b, c = s["baseline"], s["corrected"]
        tb = b["threat"] / b["n"] if b["n"] else float("nan")
        tc = c["threat"] / c["n"] if c["n"] else float("nan")
        gap = tc - tb
        frows.append((formation, gap))
        print(f"{formation:<15} {b['n']:>6} {tb:>8.1%}   {c['n']:>6} {tc:>8.1%}   {gap:>+7.1%}")

    fgaps = np.array([r[1] for r in frows])
    print(f"\nthreat_acc gap by formation: mean={fgaps.mean():+.1%} std={fgaps.std():.1%} "
         f"min={fgaps.min():+.1%} max={fgaps.max():+.1%}")

    with open(REPO / "evaluation" / "phase0_variance_diagnosis_per_pair.json", "w") as f:
        json.dump({"per_pair": [{"a": p[0][0], "b": p[0][1], "n_baseline": p[1], "threat_acc_baseline": p[2],
                                 "pair_acc_baseline": p[3], "n_corrected": p[4], "threat_acc_corrected": p[5],
                                 "pair_acc_corrected": p[6], "threat_gap": p[7]} for p in rows],
                  "per_formation": [{"formation": f[0], "threat_gap": f[1]} for f in frows]}, f, indent=2)

    print("\nsaved evaluation/phase0_variance_diagnosis_raw.json, "
         "_per_pair.json, _histories.json")
    print(f"checkpoints saved under {CKPT_ROOT}/")


if __name__ == "__main__":
    main()
