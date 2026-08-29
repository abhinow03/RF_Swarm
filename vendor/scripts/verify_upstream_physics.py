"""
AUDIT.md / V5_LOG.md: provenance guard against exactly what happened at the start of the v5
program -- Phase 0 was about to measure a "ceiling" against physics that silently did not
match what the plan assumed. This asserts, against whatever generator code THIS repo
currently points to (src/swarm_intent/formations.py + src/swarm_intent/data.py), that:

  (a) GEOMETRY: `dispersed` and `converging` do not route through one shared random branch.
      Checked two ways -- statically (the exact anti-pattern found upstream and here:
      `elif formation_type == "dispersed" or formation_type == "converging":` / `in
      ("dispersed", "converging")`, via inspect.getsource on get_formation_offsets) and
      behaviourally (same-seed outputs must differ).
  (b) ACCELERATION: velocity is not held constant for the whole trajectory. Checked
      behaviourally: generate a `noise_std=0.0` "v_shape" sequence (offsets are exactly
      constant for a non-transitioning formation, so any per-step displacement change is
      pure velocity signal, not noise or blend) and confirm the first-step and last-step
      displacement vectors are NOT bit-for-bit identical.

Fails loudly (raises AssertionError with the exact evidence, non-zero exit code) rather than
returning a boolean -- this is meant to be the first thing that runs before any Phase 0
compute, per V5_STATE.json's HALT history: two full GPU-nights were nearly spent measuring a
ceiling against unfixed physics because nothing checked this automatically beforehand.

Usage:
    python scripts/verify_upstream_physics.py
    # or, programmatically:
    from scripts.verify_upstream_physics import assert_upstream_physics_fixed
    assert_upstream_physics_fixed()
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from swarm_intent.formations import get_formation_offsets  # noqa: E402
from swarm_intent.data import generate_swarm_sequence  # noqa: E402

# The exact anti-pattern this project has now found twice (locally, and in upstream's
# pre-fix commit): dispersed and converging sharing one branch, keyed by an `or` / tuple
# membership test on both string literals in the same condition.
_SHARED_BRANCH_RE = re.compile(
    r'formation_type\s*(==\s*["\']dispersed["\']\s*(or|,)\s*.*["\']converging["\']'
    r'|in\s*\(\s*["\']dispersed["\']\s*,\s*["\']converging["\']\s*\)'
    r'|in\s*\(\s*["\']converging["\']\s*,\s*["\']dispersed["\']\s*\))',
)


def check_geometry_fix() -> None:
    source = inspect.getsource(get_formation_offsets)
    m = _SHARED_BRANCH_RE.search(source)
    if m:
        raise AssertionError(
            "GEOMETRY FIX NOT PRESENT: get_formation_offsets() still routes 'dispersed' "
            "and 'converging' through a single shared branch (matched pattern: "
            f"{m.group(0)!r}). This is the exact bug diagnosed in AUDIT.md sec AF/AG and "
            "requested in docs/UPSTREAM_ISSUES.md #1. Do not proceed to Phase 0."
        )

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    off_dispersed = get_formation_offsets("dispersed", spread=1.0, rng=rng_a)
    off_converging = get_formation_offsets("converging", spread=1.0, rng=rng_b)
    if off_dispersed.shape != (6, 3) or off_converging.shape != (6, 3):
        raise AssertionError(
            f"get_formation_offsets returned unexpected shapes: dispersed={off_dispersed.shape}, "
            f"converging={off_converging.shape} (expected (6, 3) for both)."
        )
    if np.allclose(off_dispersed, off_converging):
        raise AssertionError(
            "GEOMETRY FIX NOT PRESENT (behavioural check): get_formation_offsets('dispersed', "
            "rng=seed(42)) and get_formation_offsets('converging', rng=seed(42)) produced "
            "identical offsets -- same code path, same random draws."
        )
    print("  [PASS] geometry: dispersed/converging are not a shared branch (static + behavioural)")


def check_acceleration_fix() -> None:
    seq, labels, meta = generate_swarm_sequence(
        formation_type="v_shape", n_timesteps=50, dt=0.5, spread=1.0, noise_std=0.0,
        rng=np.random.default_rng(7),
    )
    # v_shape's offsets are constant with time and noise_std=0.0 removes the only other
    # source of per-step variation, so consecutive-timestep displacement is pure velocity*dt.
    first_step_disp = seq[1] - seq[0]
    last_step_disp = seq[-1] - seq[-2]
    if np.allclose(first_step_disp, last_step_disp, atol=1e-9):
        raise AssertionError(
            "ACCELERATION FIX NOT PRESENT: generate_swarm_sequence's velocity is constant "
            "across the whole trajectory -- the first-step and last-step displacement "
            f"vectors are identical ({first_step_disp} == {last_step_disp}, noise_std=0.0, "
            "v_shape has constant offsets, so this isolates velocity alone). Requested but "
            "never landed anywhere before the v5 plan introduced it as a precondition. "
            "Do not proceed to Phase 0."
        )
    print(f"  [PASS] acceleration: first-step displacement {first_step_disp} != "
         f"last-step displacement {last_step_disp}")


def assert_upstream_physics_fixed() -> None:
    check_geometry_fix()
    check_acceleration_fix()


def main() -> int:
    print("Verifying upstream physics fixes are present in the CURRENT repo's generator code...")
    try:
        assert_upstream_physics_fixed()
    except AssertionError as e:
        print(f"\n[FAIL] {e}\n", file=sys.stderr)
        return 1
    print("\nBoth fixes confirmed present. Safe to proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
