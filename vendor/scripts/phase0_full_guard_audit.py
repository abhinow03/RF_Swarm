"""
V5 program, Phase 0, post-guard-fix step 4: the dispersed_converging_ambiguity
guard was firing on 75.8% of windows including cases where neither class was in
the top-2 -- now fixed. This script applies the SAME methodology (fire rate, what
fraction of firings are justified by the condition the mechanism claims to test)
to every OTHER guard and reduction rule in stgt_bridge.py:

  oov_name                       -- fires when >=1 window maps to UNKNOWN_FORMATION.
  dominant_history_contradiction -- fires when the two known formations' window
                                     counts are tied.
  low_confidence                 -- fires when EVERY window is below 0.6 confidence.
  key_windows cap                -- DEFAULT_MAX_KEY_WINDOWS=10; does capping ever
                                     drop the only evidence of one side of a real
                                     transition from the narrative shown downstream?
  leading/trailing transitioning-run trim -- robust=True's _robust_reduce step;
                                     does it ever strip a window whose TRUE label
                                     was a real formation (discarding genuine
                                     signal, not noise)?

For the three boolean guards (oov_name/dominant_history_contradiction/
low_confidence), "spurious" is defined the SAME way sec AG's dispersed_converging
defect-quantification used: among trajectories where a guard is the SOLE reason
bucket != A (guard_reasons == [that guard]), does the structural rules_key (which
classify_observation computes BEFORE guard checks, and returns regardless of
bucket) already equal the ground-truth pair? If so, the guard blocked what would
otherwise have been the correct answer -- spurious by construction, not a matter
of interpretation. Where ground truth about the WINDOW itself is available
(per-timestep true labels from build_long_sequence_labeled), also reports whether
the guard's own literal justification held (e.g. for oov_name: was the offending
window's true label actually "transitioning", i.e. genuinely ambiguous, or a real
formation the classifier simply got wrong).

Same seed=999 population as every other Phase 0 measurement.

Usage (run inside tmux):
    python scripts/phase0_full_guard_audit.py --n 1000
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

from swarm_intent.coverage import classify_observation, BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402
from swarm_intent.stgt_bridge import (  # noqa: E402
    _validate_formation, _select_key_window_indices, _is_ambiguous_dispersed_converging,
    UNKNOWN_FORMATION, DEFAULT_MAX_KEY_WINDOWS, DEFAULT_ROBUST_THRESHOLD,
)
from swarm_intent.progress import Reporter  # noqa: E402

from phase0_decompose_failures import (  # noqa: E402
    sample_chain, build_long_sequence_labeled, ground_truth_pair, CLASS_ORDER,
)

DATA_DIR = REPO / "swarm_data"
CHECKPOINT = DATA_DIR / "best_model.pt"
SEED = 999
BOOLEAN_GUARDS = ("oov_name", "dominant_history_contradiction", "low_confidence",
                  "dispersed_converging_ambiguity")


def window_true_labels(predictions, true_labels, window_size=50, stride=10):
    out = []
    for w in range(len(predictions)):
        start, end = w * stride, min(w * stride + window_size, len(true_labels))
        seg = true_labels[start:end]
        out.append(Counter(seg).most_common(1)[0][0] if seg else None)
    return out


def audit_boolean_guards(records):
    """records: list of dicts with gt_pair, rules_key, bucket, guard_reasons.
    Returns {guard: {fire_rate, n_fires, n_sole_fires, n_sole_spurious,
    n_any_participation_and_correct}}."""
    n_total = len(records)
    out = {}
    for g in BOOLEAN_GUARDS:
        n_fires = sum(1 for r in records if g in r["guard_reasons"])
        sole = [r for r in records if r["guard_reasons"] == [g]]
        n_sole = len(sole)
        n_sole_spurious = sum(1 for r in sole if r["rules_key"] is not None
                              and tuple(r["rules_key"]) == tuple(r["gt_pair"]))
        any_part = [r for r in records if g in r["guard_reasons"]]
        n_any_correct = sum(1 for r in any_part if r["rules_key"] is not None
                            and tuple(r["rules_key"]) == tuple(r["gt_pair"]))
        out[g] = {
            "n_total": n_total, "n_fires": n_fires, "fire_rate": n_fires / n_total if n_total else 0.0,
            "n_sole_firing": n_sole,
            "n_sole_firing_spurious": n_sole_spurious,
            "sole_firing_spurious_rate": n_sole_spurious / n_sole if n_sole else 0.0,
            "n_any_participation_blocking_correct_answer": n_any_correct,
        }
    return out


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
    reporter = Reporter("phase0_full_guard_audit", args.n, rate_hint=8.0)

    pair_eligible_records = []  # for boolean guard audit (chain 1-2 only)
    oov_window_justification = Counter()  # "genuine_transitioning" vs "spurious_misclassification"
    cap_audit = {"n_capped": 0, "n_capped_missing_endpoint": 0, "n_total_windows_gt_cap": 0,
                "n_total_chain2plus": 0}
    trim_audit = {"n_trimmed_any": 0, "n_trimmed_windows_total": 0,
                 "n_trimmed_windows_spurious": 0, "n_pair_eligible": 0}

    for i in range(args.n):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq, true_labels = build_long_sequence_labeled(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        wtl = window_true_labels(predictions, true_labels)

        gt_pair = ground_truth_pair(chain)
        info = classify_observation(predictions, robust=False)

        if gt_pair is not None:
            pair_eligible_records.append({
                "gt_pair": list(gt_pair), "rules_key": list(info["rules_key"]) if info["rules_key"] else None,
                "bucket": info["bucket"], "guard_reasons": info["guard_reasons"],
            })

            # oov_name window-level justification: for windows STGT mapped to UNKNOWN,
            # was the window's TRUE label actually "transitioning" (genuine) or a real
            # formation the classifier simply misclassified (spurious)?
            if "oov_name" in info["guard_reasons"]:
                for pred, true_label in zip(predictions, wtl):
                    validated = _validate_formation(pred["formation_type"])
                    if validated == UNKNOWN_FORMATION:
                        if true_label == "transitioning":
                            oov_window_justification["genuine_transitioning"] += 1
                        else:
                            oov_window_justification["spurious_misclassification"] += 1

            # leading/trailing trim audit (robust=True path)
            trim_audit["n_pair_eligible"] += 1
            info_robust = classify_observation(predictions, robust=True, robust_threshold=DEFAULT_ROBUST_THRESHOLD)
            robust_info = info_robust.get("robust_recovery")
            if robust_info:
                n_lead = robust_info.get("stripped_leading", 0)
                n_trail = robust_info.get("stripped_trailing", 0)
                if n_lead or n_trail:
                    trim_audit["n_trimmed_any"] += 1
                    trimmed_idx = list(range(0, n_lead)) + list(range(len(predictions) - n_trail, len(predictions)))
                    for idx in trimmed_idx:
                        if 0 <= idx < len(wtl):
                            trim_audit["n_trimmed_windows_total"] += 1
                            if wtl[idx] != "transitioning":
                                trim_audit["n_trimmed_windows_spurious"] += 1

        # key_windows cap audit -- meaningful only when raw window count exceeds the cap
        formation_seq = [_validate_formation(p["formation_type"]) for p in predictions]
        if len(chain) >= 2:
            cap_audit["n_total_chain2plus"] += 1
        if len(predictions) > DEFAULT_MAX_KEY_WINDOWS:
            cap_audit["n_total_windows_gt_cap"] += 1
            ambiguous_flags = [_is_ambiguous_dispersed_converging(p.get("class_probabilities", {}))
                               for p in predictions]
            key_idx = _select_key_window_indices(predictions, formation_seq, ambiguous_flags,
                                                 DEFAULT_MAX_KEY_WINDOWS)
            cap_audit["n_capped"] += 1
            represented = {formation_seq[idx] for idx in key_idx if formation_seq[idx] != UNKNOWN_FORMATION}
            true_formations_present = set(chain) if gt_pair is not None else set()
            if gt_pair is not None and not true_formations_present.issubset(represented):
                cap_audit["n_capped_missing_endpoint"] += 1

        reporter.update(1, item=f"seq {i}")

    reporter.status = "done"
    reporter._write()

    boolean_results = audit_boolean_guards(pair_eligible_records)

    print("\n" + "=" * 100)
    print(f"FULL BRIDGE GUARD/RULE AUDIT (n={args.n}, pair-eligible n={len(pair_eligible_records)})")
    print("=" * 100)
    print("\n-- Boolean guards --")
    print("| guard | fire_rate | n_sole_firing | sole_firing_spurious_rate | any-participation "
         "blocking-correct |")
    print("|---|---|---|---|---|")
    for g, r in boolean_results.items():
        print(f"| {g} | {r['n_fires']}/{r['n_total']} ({r['fire_rate']:.1%}) | {r['n_sole_firing']} | "
             f"{r['n_sole_firing_spurious']}/{r['n_sole_firing']} "
             f"({r['sole_firing_spurious_rate']:.1%}) | {r['n_any_participation_blocking_correct_answer']} |")

    n_oov_total = sum(oov_window_justification.values())
    print(f"\n-- oov_name window-level justification (n_unknown_windows={n_oov_total} across all "
         f"oov_name-firing trajectories) --")
    for k, v in oov_window_justification.most_common():
        print(f"  {k}: {v}/{n_oov_total} ({v/n_oov_total:.1%})" if n_oov_total else f"  {k}: 0")

    print(f"\n-- key_windows cap (DEFAULT_MAX_KEY_WINDOWS={DEFAULT_MAX_KEY_WINDOWS}) --")
    print(f"  trajectories with raw window count > cap: {cap_audit['n_total_windows_gt_cap']}/{args.n}")
    print(f"  of those, capping actually removed windows: {cap_audit['n_capped']}")
    print(f"  of those, capped selection is MISSING at least one true endpoint formation: "
         f"{cap_audit['n_capped_missing_endpoint']}/{cap_audit['n_capped'] if cap_audit['n_capped'] else 1}")

    print(f"\n-- leading/trailing transitioning-run trim (robust=True path) --")
    print(f"  pair-eligible trajectories: {trim_audit['n_pair_eligible']}")
    print(f"  trajectories where trim removed >=1 window: {trim_audit['n_trimmed_any']} "
         f"({trim_audit['n_trimmed_any']/trim_audit['n_pair_eligible']:.1%})" if trim_audit['n_pair_eligible'] else "")
    print(f"  total windows trimmed: {trim_audit['n_trimmed_windows_total']}")
    print(f"  of those, spurious (true label was NOT \"transitioning\" -- genuine signal discarded): "
         f"{trim_audit['n_trimmed_windows_spurious']}/{trim_audit['n_trimmed_windows_total']} "
         f"({trim_audit['n_trimmed_windows_spurious']/trim_audit['n_trimmed_windows_total']:.1%})"
         if trim_audit['n_trimmed_windows_total'] else "  (no trims observed)")

    out = {"boolean_guards": boolean_results, "oov_window_justification": dict(oov_window_justification),
          "key_windows_cap": cap_audit, "trim_audit": trim_audit}
    (REPO / "evaluation" / "phase0_full_guard_audit.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved evaluation/phase0_full_guard_audit.json")


if __name__ == "__main__":
    main()
