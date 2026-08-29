"""
V5 program, Phase 0 (docs/UPSTREAM_ISSUES.md issue #3 follow-up): is chain-length-2's
18.7% pair accuracy an STGT recognition failure, or is the destination formation B
structurally invisible to the sliding-window input before STGT ever sees it?

Reuses the EXACT same seed=999/n=1000 population, sample_chain/build_long_sequence_labeled/
ground_truth_pair helpers, and window_size=50/stride=10 grid as
scripts/phase0_decompose_failures.py and scripts/phase0_chainlength_breakdown.py -- so
results here are directly comparable to (not a re-measurement replacing) docs/CEILING.md's
existing chain-2 numbers.

Observability criterion (NOT invented here -- this is the SAME per-window true-label
majority computation phase0_chainlength_breakdown.py's trace_trajectory() and
phase0_decompose_failures.py's window_true already use to score whether a window's
prediction is "correct"):

  A window's true label is the majority vote of true_labels[start:end] (Counter.most_common).
  Destination formation B is OBSERVABLE if at least one window's true-label majority is B --
  i.e. there is at least one point in the evaluation protocol's own window grid where a
  classifier reading B on that window would be scored correct. If NO window's majority is ever
  B, no classifier -- however accurate -- can produce a window read of B, because the
  evaluation protocol itself never presents B as the dominant content of any window.

Stratification (renamed OBS_CLEAR/OBS_PARTIAL/OBS_NONE, not A/B/C, to avoid collision with
coverage.py's RESOLVABLE/GUARDABLE/UNRESOLVABLE bucket vocabulary used throughout this
project):
  OBS_CLEAR   -- B is the true majority of >=2 windows (redundant, robust signal)
  OBS_PARTIAL -- B is the true majority of exactly 1 window (fragile, single point of failure)
  OBS_NONE    -- B is never the true majority of any window

Usage (run inside tmux -- ~1000 model inferences):
    python scripts/phase0_chain2_observability.py --n 1000
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

from swarm_intent.config import BASE_FORMATIONS, TRANSITION_CLASS  # noqa: E402
from swarm_intent.coverage import classify_observation, BUCKET_A  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402
from swarm_intent.stgt_bridge import _is_ambiguous_dispersed_converging  # noqa: E402

from phase0_decompose_failures import (  # noqa: E402
    sample_chain, build_long_sequence_labeled, ground_truth_pair,
)
from build_sft_dataset import RULES  # noqa: E402

DATA_DIR = REPO / "swarm_data"
CHECKPOINT = DATA_DIR / "best_model.pt"
SEED = 999
WINDOW_SIZE, STRIDE = 50, 10
N_TRACE = 20


def wilson_ci95(k: int, n: int):
    if n == 0:
        return (0.0, 0.0, 0.0)
    z = 1.959964
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def window_composition(true_labels, start, end, a, b):
    """Per-window (true_majority_label, frac_a, frac_transitioning, frac_b)."""
    seg = true_labels[start:end]
    if not seg:
        return None, 0.0, 0.0, 0.0
    c = Counter(seg)
    n = len(seg)
    majority = c.most_common(1)[0][0]
    return majority, c.get(a, 0) / n, c.get(TRANSITION_CLASS, 0) / n, c.get(b, 0) / n


def score_group(records, key):
    n = len(records)
    if n == 0:
        return {"n": 0, "pair_accuracy": None, "threat_accuracy": None}
    n_pair = n_threat = 0
    for r in records:
        gt = tuple(r["gt_pair"])
        true_threat = RULES[gt][0]
        rec = r[key]
        if rec is None:
            continue
        rec = tuple(rec)
        n_pair += rec == gt
        if rec in RULES:
            n_threat += RULES[rec][0] == true_threat
    return {"n": n, "n_pair_correct": n_pair, "pair_accuracy": n_pair / n,
            "n_threat_correct": n_threat, "threat_accuracy": n_threat / n}


def categorize_root_cause(rec):
    """Programmatic FIRST PASS at the 6-way root-cause taxonomy the user specified.
    This is a suggestion, not a final verdict -- the actual write-up manually reviews
    each traced case's full window table before reporting a category, per the explicit
    instruction not to force ambiguous cases into a category."""
    if rec["obs_group"] == "OBS_NONE":
        return "1_destination_not_observable"
    # B IS observable (>=1 window's true majority is B). Was STGT's prediction on
    # that/those window(s) actually B?
    b_windows_wrong = [w for w in rec["windows"] if w["true"] == rec["gt_pair"][1]
                       and w["pred"] != rec["gt_pair"][1]]
    b_windows_total = [w for w in rec["windows"] if w["true"] == rec["gt_pair"][1]]
    if b_windows_total and len(b_windows_wrong) == len(b_windows_total):
        return "3_stgt_misclassification"
    if b_windows_total and len(b_windows_wrong) < len(b_windows_total):
        # STGT got at least one B-majority window right, but the pipeline still failed
        return "4_bridge_reduction_issue"
    return "6_other_insufficient_evidence"


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
    print(f"checkpoint: epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f} "
         f"val_acc={ckpt.get('val_acc'):.4f}")
    model = STGTModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    rng = np.random.default_rng(args.seed)
    reporter = Reporter("phase0_chain2_observability", args.n, rate_hint=8.0)

    chain2_records = []

    for i in range(args.n):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq, true_labels = build_long_sequence_labeled(chain, rng, spread, noise_std)

        if len(chain) != 2:
            reporter.update(1, item=f"seq {i} (chain_len={len(chain)}, skip)")
            continue

        a, b = chain[0], chain[1]
        gt_pair = ground_truth_pair(chain)

        predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std,
                                               window_size=WINDOW_SIZE, stride=STRIDE, dt=0.5)

        # blend region in absolute timesteps (as ACTUALLY realized in true_labels,
        # not just the sampled blend_start/blend_end args -- reads back what the
        # eval protocol's own ground truth says the transition span is)
        trans_idx = [t for t, lab in enumerate(true_labels) if lab == TRANSITION_CLASS]
        blend_start_ts = trans_idx[0] if trans_idx else None
        blend_end_ts = trans_idx[-1] + 1 if trans_idx else None

        windows = []
        n_maj_b = 0
        for w, pred in enumerate(predictions):
            start = pred["window_start_t"]
            end = pred["window_end_t"] + 1
            majority, frac_a, frac_t, frac_b = window_composition(true_labels, start, end, a, b)
            if majority == b:
                n_maj_b += 1
            cp = pred.get("class_probabilities", {})
            ranked = sorted(cp.items(), key=lambda kv: kv[1], reverse=True)[:2]
            windows.append({
                "w": w, "start_t": start, "end_t": end,
                "true": majority, "frac_a": round(frac_a, 3), "frac_transitioning": round(frac_t, 3),
                "frac_b": round(frac_b, 3),
                "pred": pred["formation_type"], "confidence": round(pred["formation_confidence"], 4),
                "top2": [(n, round(p, 4)) for n, p in ranked],
                "dc_ambiguous": _is_ambiguous_dispersed_converging(cp),
            })

        if n_maj_b >= 2:
            obs_group = "OBS_CLEAR"
        elif n_maj_b == 1:
            obs_group = "OBS_PARTIAL"
        else:
            obs_group = "OBS_NONE"

        info_f = classify_observation(predictions, robust=False)
        info_t = classify_observation(predictions, robust=True)
        rec_f = list(info_f["rules_key"]) if info_f["bucket"] == BUCKET_A else None
        rec_t = list(info_t["rules_key"]) if info_t["bucket"] == BUCKET_A else None

        chain2_records.append({
            "i": i, "true_chain": chain, "gt_pair": list(gt_pair),
            "n_timesteps": int(long_seq.shape[0]), "n_windows": len(predictions),
            "blend_start_ts": blend_start_ts, "blend_end_ts": blend_end_ts,
            "blend_duration_ts": (blend_end_ts - blend_start_ts) if trans_idx else 0,
            "n_maj_b_windows": n_maj_b, "obs_group": obs_group,
            "windows": windows,
            "bucket_false": info_f["bucket"], "guard_reasons_false": info_f["guard_reasons"],
            "recovered_pair_false": rec_f,
            "bucket_true": info_t["bucket"], "guard_reasons_true": info_t["guard_reasons"],
            "recovered_pair_true": rec_t,
            "pair_correct_false": rec_f == list(gt_pair),
            "pair_correct_true": rec_t == list(gt_pair),
        })

        reporter.update(1, item=f"seq {i} chain2 obs={obs_group}")

    reporter.status = "done"
    reporter._write()

    n_total = len(chain2_records)
    n_obs = sum(1 for r in chain2_records if r["obs_group"] != "OBS_NONE")
    n_unobs = n_total - n_obs
    p, lo, hi = wilson_ci95(n_obs, n_total)

    print("\n" + "=" * 100)
    print("STEP 1: CHAIN-2 OBSERVABILITY")
    print("=" * 100)
    print(f"Total chain-2 trajectories: {n_total}")
    print(f"Trajectories where B is observable (>=1 window true-majority-B): {n_obs}")
    print(f"Trajectories where B is never observable: {n_unobs}")
    print(f"Percentage observable: {p:.1%} (95% CI [{lo:.1%}, {hi:.1%}])")

    obs_group_counts = Counter(r["obs_group"] for r in chain2_records)
    print(f"\nBreakdown: OBS_CLEAR={obs_group_counts['OBS_CLEAR']}, "
         f"OBS_PARTIAL={obs_group_counts['OBS_PARTIAL']}, OBS_NONE={obs_group_counts['OBS_NONE']}")

    print("\n" + "=" * 100)
    print("STEP 2: ACCURACY BY OBSERVABILITY GROUP")
    print("=" * 100)
    print("| group | n | pair_acc (F) | pair_acc (T) | threat_acc (F) | threat_acc (T) |")
    print("|---|---|---|---|---|---|")
    group_summary = {}
    for g in ("OBS_CLEAR", "OBS_PARTIAL", "OBS_NONE"):
        sub = [r for r in chain2_records if r["obs_group"] == g]
        sf = score_group(sub, "recovered_pair_false")
        st = score_group(sub, "recovered_pair_true")
        group_summary[g] = {"robust_false": sf, "robust_true": st}
        if sf["n"]:
            print(f"| {g} | {sf['n']} | {sf['pair_accuracy']:.1%} | {st['pair_accuracy']:.1%} | "
                 f"{sf['threat_accuracy']:.1%} | {st['threat_accuracy']:.1%} |")
        else:
            print(f"| {g} | 0 | n/a | n/a | n/a | n/a |")

    obs_pool = [r for r in chain2_records if r["obs_group"] != "OBS_NONE"]
    unobs_pool = [r for r in chain2_records if r["obs_group"] == "OBS_NONE"]
    s_obs = score_group(obs_pool, "recovered_pair_false")
    s_unobs = score_group(unobs_pool, "recovered_pair_false")
    print(f"\nOBSERVABLE B vs UNOBSERVABLE B (robust=False, pooled OBS_CLEAR+OBS_PARTIAL vs OBS_NONE):")
    print(f"  observable:   n={s_obs['n']}, pair_acc={s_obs['pair_accuracy']:.1%}" if s_obs['n'] else "  observable: n=0")
    print(f"  unobservable: n={s_unobs['n']}, pair_acc={s_unobs['pair_accuracy']:.1%}" if s_unobs['n'] else "  unobservable: n=0")
    if s_obs['n'] and s_unobs['n']:
        print(f"  difference: {s_obs['pair_accuracy'] - s_unobs['pair_accuracy']:+.1%}")

    # --- step 3: select 20 failing cases, balanced across observability groups ---
    failures = [r for r in chain2_records if not r["pair_correct_false"]]
    by_group = {"OBS_CLEAR": [], "OBS_PARTIAL": [], "OBS_NONE": []}
    for r in failures:
        by_group[r["obs_group"]].append(r)

    target_per_group = N_TRACE // 3
    trace_sample = []
    for g in ("OBS_CLEAR", "OBS_PARTIAL", "OBS_NONE"):
        trace_sample.extend(by_group[g][:target_per_group])
    remaining = N_TRACE - len(trace_sample)
    if remaining > 0:
        leftovers = [r for g in ("OBS_NONE", "OBS_PARTIAL", "OBS_CLEAR")
                    for r in by_group[g][target_per_group:]]
        trace_sample.extend(leftovers[:remaining])
    trace_sample = trace_sample[:N_TRACE]

    print("\n" + "=" * 100)
    print(f"STEP 3: {len(trace_sample)}-CASE FAILURE TRACE (balanced across observability groups)")
    print("=" * 100)

    trace_lines = []
    root_cause_counts = Counter()
    for r in trace_sample:
        suggested = categorize_root_cause(r)
        root_cause_counts[suggested] += 1
        lines = [f"=== trajectory {r['i']}: {r['gt_pair'][0]} -> {r['gt_pair'][1]} "
                f"(obs_group={r['obs_group']}, n_maj_b_windows={r['n_maj_b_windows']}) ==="]
        lines.append(f"n_timesteps={r['n_timesteps']}, n_windows={r['n_windows']}, "
                     f"blend=[{r['blend_start_ts']},{r['blend_end_ts']}) "
                     f"(duration={r['blend_duration_ts']} timesteps)")
        lines.append(f"{'w':>3} {'t-range':>10} {'true':>13} {'pred':>13} {'conf':>6} "
                     f"{'frac_a':>7} {'frac_t':>7} {'frac_b':>7} {'top2':>28} {'dc_amb':>7}")
        for w in r["windows"]:
            mark = " <-WRONG" if w["true"] != w["pred"] else ""
            top2_str = ",".join(f"{n}={p:.2f}" for n, p in w["top2"])
            lines.append(f"{w['w']:>3} {w['start_t']:>4}-{w['end_t']:>4} {str(w['true']):>13} "
                        f"{w['pred']:>13} {w['confidence']:>6.2f} {w['frac_a']:>7.2f} "
                        f"{w['frac_transitioning']:>7.2f} {w['frac_b']:>7.2f} {top2_str:>28} "
                        f"{str(w['dc_ambiguous']):>7}{mark}")
        lines.append(f"bucket(robust=False)={r['bucket_false']}, guards={r['guard_reasons_false']}, "
                     f"recovered={r['recovered_pair_false']}")
        lines.append(f"bucket(robust=True)={r['bucket_true']}, guards={r['guard_reasons_true']}, "
                     f"recovered={r['recovered_pair_true']}")
        lines.append(f"SUGGESTED root cause (programmatic first pass, needs manual review): {suggested}")
        trace_lines.append("\n".join(lines))

    trace_path = REPO / "evaluation" / "phase0_chain2_observability_trace.txt"
    trace_path.write_text("\n\n".join(trace_lines))
    print(f"\nsaved full trace to {trace_path}")
    print(f"\nSUGGESTED root-cause distribution (programmatic first pass, {len(trace_sample)} cases):")
    for cat, c in root_cause_counts.most_common():
        print(f"  {cat}: {c}/{len(trace_sample)} ({c/len(trace_sample):.1%})")

    out = {
        "n": args.n, "seed": args.seed, "n_chain2": n_total,
        "n_observable": n_obs, "n_unobservable": n_unobs,
        "pct_observable": p, "pct_observable_ci95": [lo, hi],
        "obs_group_counts": dict(obs_group_counts),
        "group_summary": group_summary,
        "observable_vs_unobservable": {"observable": s_obs, "unobservable": s_unobs},
        "trace_sample_indices": [r["i"] for r in trace_sample],
        "suggested_root_cause_distribution": dict(root_cause_counts),
        "records": chain2_records,
    }
    out_path = REPO / "evaluation" / "phase0_chain2_observability.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
