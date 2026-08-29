"""
Step 2 of the "settle delta_v, recalibrate synth_context, diagnose the prior
skew" session (AUDIT.md sec W/continued).

Exports a compact, COMMITTED empirical-percentile snapshot of the real
regression-label distributions (sec V, swarm_data/_reg_distribution_analysis.npz,
n=5879) as Python literals for llm_finetuning/build_sft_dataset.py to sample
from. Deliberately NOT a runtime dependency on swarm_data/ (gitignored,
teammate-provided, not present on a fresh clone/CI) -- 101-point (1% steps)
empirical-CDF breakpoints per field are close enough to true empirical
sampling for narrative-text purposes and small enough to commit as literals.

Re-run this and paste the printed dict into build_sft_dataset.py's
REAL_REG_PERCENTILES constant if swarm_data/_reg_distribution_analysis.npz is
regenerated (e.g. after a further STGT retrain).

Usage:
    python scripts/export_real_reg_percentiles.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIELDS = ("velocity_physical", "approach_rate", "stability", "delta_v_physical", "delta_stability")


def main():
    d = np.load(REPO / "swarm_data" / "_reg_distribution_analysis.npz")
    print("REAL_REG_PERCENTILES = {  # AUDIT.md sec V, n=5879, 1%-step empirical CDF breakpoints")
    for field in FIELDS:
        vals = np.percentile(d[field], np.arange(0, 101))
        rounded = [round(float(v), 4) for v in vals]
        print(f"    {field!r}: {rounded!r},")
    print("}")


if __name__ == "__main__":
    main()
