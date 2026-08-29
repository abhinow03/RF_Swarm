# Run selection notes

## Run A formation: diamond, not encirclement (2026-08-28)

The spec originally proposed encirclement for Run A ("nominal, everything works,
high confidence"). Empirically tested first: over a genuinely fresh 180s capture
of `gate_k6` (encirclement), the real STGT classifier landed on "dispersed" 51%
of the time (167/327 messages), oscillating across 5 different labels. K̂=6 and
position RMSE (27m) were both within the pre-registered NOISE_MODEL.md §4a range
— only the formation call was unstable.

This is not a bug. It's consistent with the already-documented Phase-5 finding
(carried in project memory) that encirclement itself is a marginal case under
real-EKF tracking (~44% correct), not one of the clean cases — v_shape was the
only formation flagged as "known-hard" going into this, but encirclement turns
out to sit in the same tier for this array geometry/scenario.

Switched to `gate_k6_diamond` and re-tested the same way: 100% stable across 344
formation messages, 99.1% confidence, K̂=6, formation_correct=true, RMSE 29m.
Locked in as Run A (`runs/runA_ble_nominal/`).

Takeaway for anyone extending this: don't assume a formation is "the easy one"
from the STGT class list alone — verify empirically against real EKF tracks over
a multi-minute window (single-snapshot captures under-sample the classifier's
real variance) before locking a formation into a demo run's story.
