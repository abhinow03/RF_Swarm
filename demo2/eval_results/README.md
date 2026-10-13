# Evaluation results — how error accumulates through the pipeline

Four scripts, run fresh on 2026-10-13 (one month after the capstone review),
scoring every estimator stage against ground truth. Raw output is in the
sibling files; this doc is the synthesis — what the numbers mean and how
error at one stage propagates into the next.

| File | What it measures | How to reproduce |
|---|---|---|
| `ekf_validation.txt` | TDoA+EKF localization RMSE/NEES, offline, 8 clock-noise seeds | `python3 capstone_ws/src/capstone_track/scripts/validate_ekf.py` |
| `formation_confusion_truth.txt` | STGT formation accuracy fed CLEAN truth trajectories (isolates the classifier from upstream noise) | `python3 capstone_ws/src/capstone_formation/scripts/gate_confusion.py` |
| `formation_confusion_live.jsonl` | STGT formation accuracy fed REAL tracks (real BLE IQ → RFFP → blind clustering → TDoA+EKF → STGT), one line per scenario | live pipeline + `gate_confusion_live.py <formation> <duration> <results_file>`, once per `gate_k6*` scenario |
| `cv_baseline_rmse.txt` | Constant-velocity path-prediction error, bucketed by prediction horizon | live pipeline + `cv_baseline_eval.py <duration>` during `gate_k6` |

## The error chain, stage by stage

```
real BLE IQ
   |  (RFFP router + frozen encoder — identity signal only, no position info)
   v
blind embeddings --------------------------------- no position error introduced here
   |  (eigengap clustering: K estimation + per-burst association)
   v
per-emitter ID (K=6, bijective)  ------------------ clustering is correct in every run below;
   |                                                 this stage does NOT explain the position error
   v
TDoA + iterated EKF  ------------------------------ POSITION ERROR ORIGINATES HERE
   |   mean RMSE 17.07m across 8 clock-noise seeds (offline, validate_ekf.py)
   |   range 14.36m–26.72m depending on the clock-noise draw
   |   root cause: per-receiver clock bias is only weakly observable from a
   |   compact swarm's TDoA alone (proven by SVD in Phase 4) — the filter
   |   leans on a 30ns disciplined-array PRIOR instead of a clean measurement,
   |   so position error is dominated by how far that prior's realization sits
   |   from the true clock, not by sensor noise on any single burst.
   v
tracks (position + velocity + covariance)
   |                        |
   |                        v
   |                  CV-baseline path prediction
   |                  OVERALL RMSE 36.64m (live), rising from 34.27m at the
   |                  0s horizon to 39.34m at the 5s horizon (cv_baseline_rmse.txt)
   |                  = EKF position error (above) + CV extrapolation error,
   |                    and the extrapolation term visibly GROWS with horizon
   |                    (+5.07m from 0s -> 5s) — this is compounding, not
   |                    constant, error: a wrong velocity estimate's position
   |                    impact scales with how far ahead you extrapolate it.
   v
STGT formation classification
   |   TRUTH-fed (no EKF error in the input): 5/5 correct, conf 0.84-0.995
   |   REAL-fed (full chain above feeds the window): 3/5 correct
   |     - column:    correct, conf 0.984
   |     - diamond:   correct, conf 0.991
   |     - dispersed: correct, conf 0.995
   |     - v_shape:        MISCLASSIFIED as diamond, conf 0.978
   |     - encirclement:   MISCLASSIFIED as diamond, conf 0.989
   v
(narration, AR bridge — not re-scored here; they consume /formation and
 /predictions downstream of all of the above, so they inherit this error
 chain rather than adding their own)
```

## The headline finding: where the 2/5 formation misses actually come from

The truth-fed and real-fed confusion matrices isolate the question "is this
a classifier bug, or an upstream-noise problem?" — **it's upstream noise,**
not the classifier:

- The classifier itself is 5/5 on clean geometry, including v_shape and
  encirclement, both at confidence >0.83.
- Fed real, EKF-estimated tracks (this session's ~17-36m position/path
  error), v_shape and encirclement collapse into diamond — both at >0.97
  confidence, i.e. the classifier isn't uncertain, it's **confidently wrong**
  because the geometry it's looking at has been distorted by upstream error
  into something that really does look like a diamond at that noise level.
- Column, diamond, and dispersed survive the same noise untouched.

This matches and sharpens a pre-registered Phase-5 finding (the project's
`NOISE_MODEL.md` and prior gate runs already flagged v_shape/encirclement as
the hard/marginal classes): this run shows the specific failure mode is a
confident collapse into a neighboring, more symmetric class, not a vague
low-confidence "I don't know."

## An honest discrepancy, also worth reporting

This session's live numbers (36.64m path RMSE) are meaningfully worse than
figures recorded in earlier project documentation (`demo/README.md` cites
~15m EKF RMSE from an earlier real run; project notes from Phase 5 cite CV
path RMSE ~12.95m). The offline `validate_ekf.py` gate run in this same
session still reproduces the originally-documented ~17m mean — so the
*estimator code* is unchanged and behaves as designed. The gap shows up only
in the **live** path, which also depends on random clock-noise/burst-timing
draws for this particular session and on the genuinely slower RFF pre-encode
observed this run (~145s vs. the ~2-3s recorded in earlier sessions — see
below). Whether that's pure noise-draw variance or a symptom of something
slower/different in this month-later environment (GPU contention, driver
drift) isn't disentangled here — flagging it rather than quietly reporting
only the better-looking offline number.

**Also flagged:** `capstone_rff`'s GPU pre-encode step took ~143-146s in
every scenario this session, versus ~2-3s recorded in prior sessions despite
the GPU being free and CUDA available. Not investigated further here since
it didn't block getting results, but worth a look before relying on fast
pre-encode timing assumptions elsewhere (e.g. `capstone_narrate`'s startup
guard logic).
