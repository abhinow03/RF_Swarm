"""
Step 2 of the "settle delta_v, recalibrate synth_context, diagnose the prior
skew" session (AUDIT.md sec W/continued): before/after narrative-field
proportions from recalibrating synth_context() (llm_finetuning/build_sft_dataset.py)
to REAL_REG_PERCENTILES (AUDIT.md sec V, n=5879) instead of hand-picked uniforms.

"Before" reimplements the OLD sampling formulas standalone (not by reverting the
file) purely to compute a comparison distribution -- the OLD synth_context() is
gone from the codebase on purpose (recalibrated, not kept as an option).

Usage:
    python scripts/report_synth_context_recalibration.py
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_finetuning"))

from swarm_intent.config import BASE_FORMATIONS  # noqa: E402
from swarm_intent import context_spec as spec  # noqa: E402

from build_sft_dataset import synth_context, REAL_REG_PERCENTILES  # noqa: E402

N_SAMPLES = 5000


def old_narrative_fields(rng: random.Random):
    """Verbatim reimplementation of the pre-recalibration sampling formulas
    (the ones this session replaced), for comparison purposes only."""
    stab_early = round(rng.uniform(0.5, 0.98), 2)
    stab_late = round(rng.uniform(0.5, 0.98), 2)
    approach = round(rng.uniform(-1.5, 1.5), 3)
    delta_v = round(rng.uniform(-1.0, 2.0), 2)
    vel_trend = (spec.VELOCITY_ACCELERATING if delta_v > 0.5
                else spec.VELOCITY_DECELERATING if delta_v < -0.5 else spec.VELOCITY_STEADY)
    if stab_late < stab_early - 0.1:
        stab_trend = spec.STABILITY_DEGRADING
    elif stab_late > stab_early + 0.1:
        stab_trend = spec.STABILITY_IMPROVING
    else:
        stab_trend = spec.STABILITY_HOLDING
    if approach < -0.1:
        spread_trend = spec.SPREAD_CONVERGING
    elif approach > 0.1:
        spread_trend = spec.SPREAD_DISPERSING
    else:
        spread_trend = spec.SPREAD_STABLE
    return vel_trend, stab_trend, spread_trend


def new_narrative_fields(rng: random.Random):
    form_a, form_b = rng.choice(BASE_FORMATIONS), rng.choice(BASE_FORMATIONS)
    ctx, _ = synth_context(form_a, form_b, rng)
    lines = {line.split(":")[0]: line for line in ctx.splitlines()}
    vel_line = lines["Velocity trend"]
    stab_line = lines["Formation stability"]
    spread_line = lines["Spread dynamics"]
    vel_trend = vel_line.split(": ", 1)[1].split(" (")[0]
    stab_trend = stab_line.split(": ", 1)[1].split(" (")[0]
    spread_trend = spread_line.split(": ", 1)[1].split(" (mean")[0]
    return vel_trend, stab_trend, spread_trend


def proportions(counter: Counter, total: int) -> dict:
    return {k: round(100 * v / total, 1) for k, v in counter.items()}


def main():
    rng_old = random.Random(0)
    rng_new = random.Random(0)

    old_vel, old_stab, old_spread = Counter(), Counter(), Counter()
    new_vel, new_stab, new_spread = Counter(), Counter(), Counter()

    for _ in range(N_SAMPLES):
        v, s, sp = old_narrative_fields(rng_old)
        old_vel[v] += 1; old_stab[s] += 1; old_spread[sp] += 1
        v, s, sp = new_narrative_fields(rng_new)
        new_vel[v] += 1; new_stab[s] += 1; new_spread[sp] += 1

    print(f"n={N_SAMPLES} synth_context() draws\n")
    for label, old_c, new_c in [
        ("velocity_trend", old_vel, new_vel),
        ("stability_trend", old_stab, new_stab),
        ("spread_dynamics", old_spread, new_spread),
    ]:
        print(f"=== {label} ===")
        print(f"  before: {proportions(old_c, N_SAMPLES)}")
        print(f"  after:  {proportions(new_c, N_SAMPLES)}")
        print()

    print("=== real population rates (AUDIT.md sec V, n=5879) for comparison ===")
    print("  spread_dynamics: converging=29.73%  dispersing=9.66%  stable=60.60%")
    print("  velocity_trend (delta_v +-0.5):     accelerating+decelerating=33.24%  steady=66.76%")
    print("  stability_trend (+-0.1 early/late): degrading+improving=63.02%  holding=36.98%")


if __name__ == "__main__":
    main()
