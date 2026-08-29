"""
Rule-0 follow-up, step 3: trajectory-by-trajectory diff of the 3 persisted seed=999
populations (eval_data/locked_seed999_*.json, produced by
scripts/rule0_2b_regenerate_populations.py). For each pair, reports:
  - byte-identical / identical-but-reordered / genuinely different
  - count of trajectories whose chain (formation sequence) differs
  - count of trajectories whose chain MATCHES but geometry (positions) differs
  - whether divergence is generator-logic (chain/timing shape) or incidental (float noise)

Usage:
    python scripts/rule0_2b_diff_populations.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "eval_data"


def load(version_name):
    path = DATA_DIR / f"locked_seed999_{version_name}.json"
    return json.loads(path.read_text())


def sha256_of_file(version_name):
    path = DATA_DIR / f"locked_seed999_{version_name}.json"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def diff_pair(name_a, name_b):
    a, b = load(name_a), load(name_b)
    ra, rb = a["records"], b["records"]

    sha_a, sha_b = sha256_of_file(name_a), sha256_of_file(name_b)
    byte_identical = sha_a == sha_b

    n = min(len(ra), len(rb))
    n_chain_diff = 0
    n_same_chain_diff_geometry = 0
    n_same_chain_diff_timesteps = 0
    n_fully_identical = 0
    first_chain_diff_i = None

    for i in range(n):
        rec_a, rec_b = ra[i], rb[i]
        if rec_a["chain"] != rec_b["chain"]:
            n_chain_diff += 1
            if first_chain_diff_i is None:
                first_chain_diff_i = i
            continue
        if rec_a["n_timesteps"] != rec_b["n_timesteps"]:
            n_same_chain_diff_timesteps += 1
            continue
        pos_a = np.array(rec_a["positions"])
        pos_b = np.array(rec_b["positions"])
        if not np.allclose(pos_a, pos_b, atol=1e-5):
            n_same_chain_diff_geometry += 1
        else:
            n_fully_identical += 1

    return {
        "pair": f"{name_a} vs {name_b}",
        "byte_identical": byte_identical,
        "sha256_a": sha_a,
        "sha256_b": sha_b,
        "n_compared": n,
        "n_fully_identical_trajectories": n_fully_identical,
        "n_chain_differs": n_chain_diff,
        "n_same_chain_different_timestep_count": n_same_chain_diff_timesteps,
        "n_same_chain_same_length_different_geometry": n_same_chain_diff_geometry,
        "first_chain_divergence_index": first_chain_diff_i,
        "classification": classify(byte_identical, n_chain_diff, n_same_chain_diff_timesteps,
                                   n_same_chain_diff_geometry, n),
    }


def classify(byte_identical, n_chain_diff, n_ts_diff, n_geom_diff, n_total):
    if byte_identical:
        return "BYTE-IDENTICAL"
    if n_chain_diff == 0 and n_ts_diff == 0 and n_geom_diff == 0:
        return "IDENTICAL-CONTENT (reordered or float-formatting only -- verify separately)"
    if n_chain_diff > 0:
        pct = 100 * n_chain_diff / n_total
        return (f"GENUINELY DIFFERENT POPULATION -- generator-logic divergence: "
               f"{n_chain_diff}/{n_total} ({pct:.1f}%) trajectories have a different formation "
               f"chain entirely (RNG stream diverged, not incidental)")
    if n_geom_diff > 0 or n_ts_diff > 0:
        return (f"SAME chains, DIFFERENT geometry/timing -- {n_geom_diff} geometry + "
               f"{n_ts_diff} timestep-count divergences (generator-logic, not incidental)")
    return "UNCLASSIFIED"


def main():
    versions = ["OLD_pre_dwell_fix_b680c80_628dc56", "MID_post_consolidation_3591051",
               "CURRENT_post_symmetrize_9061392_HEAD"]
    pairs = [(versions[0], versions[1]), (versions[1], versions[2]), (versions[0], versions[2])]

    results = [diff_pair(a, b) for a, b in pairs]

    print("=" * 100)
    print("STEP 3: TRAJECTORY-BY-TRAJECTORY DIFF, seed=999, n=1000, all 3 population versions")
    print("=" * 100)
    for r in results:
        print(f"\n--- {r['pair']} ---")
        print(f"  byte_identical: {r['byte_identical']}")
        print(f"  n_compared: {r['n_compared']}")
        print(f"  chain differs: {r['n_chain_differs']}")
        print(f"  same-chain, different timestep count: {r['n_same_chain_different_timestep_count']}")
        print(f"  same-chain, same length, different geometry: {r['n_same_chain_same_length_different_geometry']}")
        print(f"  fully identical trajectories: {r['n_fully_identical_trajectories']}")
        print(f"  first chain divergence at index: {r['first_chain_divergence_index']}")
        print(f"  CLASSIFICATION: {r['classification']}")

    out_path = REPO / "evaluation" / "rule0_2b_population_diff_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
