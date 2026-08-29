"""
Rule-0 follow-up, step 1: inventory every script version that has EVER produced a seed=999
ceiling figure cited in V5_LOG.md/CEILING.md/AUDIT.md/PREREGISTRATION.md. Fully programmatic --
every (file, commit, figure) triple below is derived by running `git log`/`git show` against
the actual repo history, not hand-transcribed from the docs.

Method: the population-GENERATING code (sample_chain/build_long_sequence_labeled -- what
determines which 1000 trajectories seed=999 draws) has exactly 3 distinct states in this
repo's history (confirmed via `git log --follow` on every file that has ever carried a copy of
it); every persisted ceiling output file is attributed to ONE of those 3 states by finding the
commit that first added that output file and checking whether it predates or postdates the
population-changing commits.

Usage:
    python scripts/rule0_2b_step1_inventory.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The 3 population-code states, in chronological order, with the commits that define each
# boundary (see scripts/rule0_2b_regenerate_populations.py's docstring for the full derivation).
POPULATION_CODE_COMMITS = {
    "OLD_pre_dwell_fix": {"introduced_by": "b680c80", "superseded_by": "3591051", "note": "inline copy in scripts/phase0_ceiling.py, unchanged b680c80..628dc56"},
    "MID_post_consolidation_pre_symmetrize": {"introduced_by": "3591051", "superseded_by": "9061392", "note": "src/swarm_intent/eval_trajectories.py, LEAD_IN_RANGE=(15,35)"},
    "CURRENT_post_symmetrize": {"introduced_by": "9061392", "superseded_by": None, "note": "src/swarm_intent/eval_trajectories.py at HEAD, LEAD_IN_RANGE=(30,50)"},
}

# Every persisted seed=999 ceiling output file this project has ever produced.
CEILING_OUTPUT_FILES = [
    "phase0_ceiling.json", "phase0_ceiling_v2.json", "phase0_ceiling_v3.json",
    "phase0_ceiling_v4.json", "phase0_ceiling_v4_robust.json", "phase0_ceiling_v5.json",
    "phase0_ceiling_v5_domfix.json", "phase0_ceiling_v5_guardfix.json",
    "phase0_ceiling_v5_oovfix.json", "phase0_ceiling_v5_trimfix.json", "phase0_ceiling_v6.json",
    "phase0_threat_ceiling_v5.json", "phase0_threat_ceiling_v5_domfix.json",
    "phase0_threat_ceiling_v5_guardfix.json", "phase0_threat_ceiling_v5_oovfix.json",
    "phase0_threat_ceiling_v5_trimfix.json", "phase0_threat_ceiling_v6.json",
]


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def commit_date(rev):
    return git("log", "-1", "--format=%ad", "--date=iso-strict", rev)


def commit_subject(rev):
    return git("log", "-1", "--format=%s", rev)


def first_add_commit(relpath):
    out = git("log", "--oneline", "--diff-filter=A", "--", relpath)
    lines = [l for l in out.splitlines() if l.strip()]
    return lines[-1].split(" ", 1)[0] if lines else None


def attribute_population(commit_hash):
    """Which of the 3 population-code states was active when `commit_hash` landed."""
    boundary_3591051_date = commit_date("3591051")
    boundary_9061392_date = commit_date("9061392")
    this_date = commit_date(commit_hash)
    if this_date < boundary_3591051_date:
        return "OLD_pre_dwell_fix"
    if this_date < boundary_9061392_date:
        return "MID_post_consolidation_pre_symmetrize"
    return "CURRENT_post_symmetrize"


def main():
    rows = []
    for fname in CEILING_OUTPUT_FILES:
        relpath = f"evaluation/{fname}"
        commit = first_add_commit(relpath)
        if commit is None:
            rows.append({"output_file": relpath, "status": "NOT FOUND IN GIT HISTORY (untracked or deleted)"})
            continue
        rows.append({
            "output_file": relpath,
            "commit": commit,
            "commit_date": commit_date(commit),
            "commit_subject": commit_subject(commit),
            "population_code_state": attribute_population(commit),
        })

    # cross-check: the file that carried the population-generating function itself, per state
    lineage = {
        "OLD_pre_dwell_fix": {
            "carrier_file": "scripts/phase0_ceiling.py (inline copy)",
            "first_commit": "b680c80", "first_commit_date": commit_date("b680c80"),
            "last_unchanged_commit": "628dc56", "last_unchanged_date": commit_date("628dc56"),
            "superseded_by_commit": "3591051", "superseded_at": commit_date("3591051"),
        },
        "MID_post_consolidation_pre_symmetrize": {
            "carrier_file": "src/swarm_intent/eval_trajectories.py",
            "introduced_by_commit": "3591051", "introduced_at": commit_date("3591051"),
            "superseded_by_commit": "9061392", "superseded_at": commit_date("9061392"),
        },
        "CURRENT_post_symmetrize": {
            "carrier_file": "src/swarm_intent/eval_trajectories.py",
            "introduced_by_commit": "9061392", "introduced_at": commit_date("9061392"),
            "superseded_by_commit": None,
            "backs_headline_figures": [
                "step26c chain-2 pair 39.9%->65.8%/threat 72.1%->76.3% (commit b411c96) -- NO persisted output JSON, prose-only",
                "CEILING.md 'Current state, 2026-08-10': chain-1 87.6% threat/85.8% pair, "
                "chain-2 77.2% threat/66.7% pair (robust=True) -- NO persisted output JSON, prose-only",
                "CEILING.md pooled bridge figure 83.0% threat/77.3% pair -- NO persisted output JSON, prose-only",
            ],
        },
    }

    result = {"population_code_lineage": lineage, "ceiling_output_files": rows}
    out_path = REPO / "evaluation" / "rule0_2b_step1_inventory.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
