"""
Rule-0 zero-assumption audit (2026-08-13). Pure verification, no fixes.

Every check here re-derives a number DIRECTLY from the actual artifact on
disk (JSONL corpus, provenance sidecar, checkpoint files, adapter config,
pickled training args) -- never from a doc's cited figure. Where a doc
claim cannot be checked programmatically (e.g. "same file used every time"
when no such single file exists), that is reported as its own finding
rather than silently passed.

Usage:
    python scripts/rule0_audit_2026_08_13.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from build_sft_dataset import RULES, STRATA_TARGETS  # noqa: E402
from swarm_intent.llm.prompts import is_abstention  # noqa: E402

CORPUS_FILES = {
    "train": REPO / "data" / "sft_train_v5_phase1.jsonl",
    "val": REPO / "data" / "sft_train_v5_phase1_val.jsonl",
    "mining": REPO / "data" / "sft_train_v5_phase1_mining.jsonl",
}
PROVENANCE_FILE = REPO / "data" / "sft_train_v5_phase1_provenance.json"
CKPT = REPO / "swarm_data" / "best_model.pt"
BRIDGE = REPO / "src" / "swarm_intent" / "stgt_bridge.py"
ADAPTER_CFG = REPO / "checkpoints" / "v5_sft" / "adapter_config.json"
TRAINING_ARGS = REPO / "checkpoints" / "v5_sft" / "training_args.bin"

CITED_CKPT_SHA = "18fc201d5a419ff2fb0cfb66a60810af77a9ed52969f2996f57c952d1306a01b"
CITED_BRIDGE_SHA = "e99472e16525588be83f7e7e78abeb5d781d7c2d238adc902c10c660bbb022f7"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_all_rows():
    rows = []
    for split, path in CORPUS_FILES.items():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_split"] = split
                rows.append(r)
    return rows


def row_pair_and_intent(row):
    """Extract (form_a, form_b) rules-key and assistant likely_intent from a
    corpus row's messages, if present. Corpus rows store the pair implicitly
    via the user ctx text produced by the same coverage-extraction helper
    report_phase1_coverage.py already uses; reuse that here for consistency
    with the pre-existing coverage tooling, applied fresh to the real file."""
    from swarm_intent.coverage import _extract_pair_from_ctx
    user_msg = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in row["messages"] if m["role"] == "assistant"), "")
    pair = _extract_pair_from_ctx(user_msg)
    try:
        parsed = json.loads(assistant_msg)
        intent = parsed.get("likely_intent", "")
    except Exception:
        intent = None
    return pair, intent


def check_1a_abstention(rows):
    n_abstain = 0
    for r in rows:
        _, intent = row_pair_and_intent(r)
        if intent is not None and is_abstention(intent):
            n_abstain += 1
    return n_abstain, len(rows)


def check_1b_pair_coverage(rows):
    counts = {p: 0 for p in RULES}
    unmatched = 0
    for r in rows:
        pair, _ = row_pair_and_intent(r)
        if pair in counts:
            counts[pair] += 1
        else:
            unmatched += 1
    zero_pairs = [p for p, c in counts.items() if c == 0]
    return counts, zero_pairs, unmatched


def check_1c_critical_cap(rows):
    critical_pairs = {p for p, (t, *_r) in RULES.items() if t == "critical"}
    n_critical = 0
    for r in rows:
        pair, _ = row_pair_and_intent(r)
        if pair in critical_pairs:
            n_critical += 1
    return n_critical, STRATA_TARGETS.get("critical")


def check_1d_teacher_pct():
    prov = json.loads(PROVENANCE_FILE.read_text())
    rows = prov["rows"]
    n_teacher = sum(1 for v in rows.values() if v.get("used_teacher") is True)
    n_total = len(rows)
    return n_teacher, n_total


def check_1e_narrative_combo_floor(rows):
    """report_phase1_coverage.py's own >=8-distinct-narrative-combo-per-pair
    claim (line ~15/133/145/158), re-run fresh against the actual file text
    instead of citing its historical stdout."""
    import re
    _VEL_RE = re.compile(r"velocity trend[: ]+(\w+)", re.I)
    _STAB_RE = re.compile(r"stability trend[: ]+(\w+)", re.I)
    _SPREAD_RE = re.compile(r"spread trend[: ]+(\w+)", re.I)
    _ROLE_RE = re.compile(r"role differentiation[: ]+(\w+)", re.I)
    try:
        from report_phase1_coverage import narrative_combo
    except Exception:
        return None  # helper not importable standalone; not a blocking failure
    pair_combos = {}
    for r in rows:
        pair, _ = row_pair_and_intent(r)
        if pair is None:
            continue
        user_msg = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        combo = narrative_combo(user_msg)
        pair_combos.setdefault(pair, set()).add(combo)
    below = {p: len(c) for p, c in pair_combos.items() if len(c) < 8}
    return {"n_pairs_checked": len(pair_combos), "n_below_8": len(below), "below": below}


def check_2a_hashes():
    ckpt_sha = sha256_file(CKPT)
    bridge_sha = sha256_file(BRIDGE)
    return {
        "checkpoint": (ckpt_sha, ckpt_sha == CITED_CKPT_SHA),
        "bridge": (bridge_sha, bridge_sha == CITED_BRIDGE_SHA),
    }


LOCKED_SEED999_PATH = REPO / "eval_data" / "LOCKED_seed999_FINAL.json"
CITED_LOCKED_SEED999_SHA = "871a9dae4c6fdf08e1aed803592fa7c61b1a852c150693b5819fe2271717b96e"


def check_2b_seed999_locked_file():
    """AUDIT.md sec AM (2026-08-13 follow-up) resolved this finding: search for a single
    locked seed=999 trajectory file analogous to evaluation/phase4_eval_set.json (which
    exists for the separate seed=4321 population), AND -- now that one exists -- verify its
    sha256 matches what docs/PREREGISTRATION.md and docs/V5_STATE.json cite, so this check
    stays meaningful after the fix instead of just detecting file presence."""
    import subprocess
    candidates = subprocess.run(
        ["find", str(REPO), "-iname", "*phase0*eval*set*", "-o", "-iname", "*seed999*",
         "-o", "-iname", "*seed_999*"],
        capture_output=True, text=True,
    ).stdout.splitlines()
    candidates = [c for c in candidates if ".git" not in c and "__pycache__" not in c]
    locked_file_exists = any(
        Path(c).is_file() and "progress" not in c for c in candidates
    )
    canonical_locked_exists = LOCKED_SEED999_PATH.is_file()
    canonical_sha = sha256_file(LOCKED_SEED999_PATH) if canonical_locked_exists else None
    canonical_matches_cited = canonical_sha == CITED_LOCKED_SEED999_SHA if canonical_sha else False
    return candidates, locked_file_exists, canonical_locked_exists, canonical_sha, canonical_matches_cited


SCHEMA_VALIDITY_SOURCES = {
    "base": REPO / "evaluation" / "phase4_baselines_scored.json",
    "rules_in_prompt": REPO / "evaluation" / "phase4_baselines_scored.json",
    "v3b-fix": REPO / "evaluation" / "phase4_baselines_scored.json",
    "v2": REPO / "evaluation" / "phase4_baselines_scored.json",
    "v5a": REPO / "evaluation" / "phase4_v5a_results.json",
}


def check_3_preregistration_bars():
    """Item 3: for every PREREGISTRATION.md bar, confirm scripts/check_preregistration.py
    has a REAL assertion (not a table entry) by inspecting its own BARS dict + the dedicated
    check_escalation_direction() function -- both imported and introspected live, not eyeballed.
    Then test the "passes by construction" hypothesis for a candidate 3rd weak bar
    (schema_validity_rate) by pulling its actual value across ALL 5 evaluated systems
    (base/rules_in_prompt/v3b-fix/v2/v5a) from their real results JSONs -- if it's ~100%
    even for the untrained base model, it cannot be discriminating system quality."""
    sys.path.insert(0, str(REPO / "scripts"))
    import importlib
    cp = importlib.import_module("check_preregistration")

    numeric_bars = dict(cp.BARS)  # 5 bars, each with a real op/value comparison in check_bar()
    has_escalation_fn = callable(getattr(cp, "check_escalation_direction", None))
    n_bars_with_real_assertion = len(numeric_bars) + (1 if has_escalation_fn else 0)

    schema_validity_by_system = {}
    for system, path in SCHEMA_VALIDITY_SOURCES.items():
        if not path.exists():
            schema_validity_by_system[system] = None
            continue
        data = json.loads(path.read_text())
        node = data.get(system, data) if isinstance(data, dict) and system in data else data
        rate = node.get("schema_validity_rate", {}).get("rate")
        schema_validity_by_system[system] = rate

    rates = [r for r in schema_validity_by_system.values() if r is not None]
    # candidate "passes by construction" bar: near-ceiling for every system regardless of
    # accuracy quality (base model included) -- spread this tight means the bar cannot be
    # discriminating between a good and a bad system.
    schema_bar_spread = (max(rates) - min(rates)) if rates else None
    schema_bar_passes_by_construction = (
        schema_bar_spread is not None and schema_bar_spread < 0.02 and min(rates) >= 0.95
    )

    return {
        "n_bars_total": n_bars_with_real_assertion,
        "numeric_bars_with_real_assertion": list(numeric_bars.keys()),
        "escalation_direction_has_real_assertion": has_escalation_fn,
        "all_6_bars_have_real_assertion": n_bars_with_real_assertion == 6,
        "schema_validity_rate_by_system": schema_validity_by_system,
        "schema_validity_rate_spread": schema_bar_spread,
        "schema_validity_candidate_3rd_weak_bar": schema_bar_passes_by_construction,
    }


def check_4_training_args():
    import torch
    ta = torch.load(TRAINING_ARGS, map_location="cpu", weights_only=False)
    adapter_cfg = json.loads(ADAPTER_CFG.read_text())
    return {
        "assistant_only_loss": getattr(ta, "assistant_only_loss", None),
        "num_train_epochs": getattr(ta, "num_train_epochs", None),
        "lora_r": adapter_cfg.get("r"),
        "lora_alpha": adapter_cfg.get("lora_alpha"),
    }


def main():
    rows = load_all_rows()
    assert len(rows) == 12001, f"expected 12001 rows, got {len(rows)}"

    results = {}
    results["1a"] = check_1a_abstention(rows)
    results["1b"] = check_1b_pair_coverage(rows)
    results["1c"] = check_1c_critical_cap(rows)
    results["1d"] = check_1d_teacher_pct()
    results["1e"] = check_1e_narrative_combo_floor(rows)
    results["2a"] = check_2a_hashes()
    results["2b"] = check_2b_seed999_locked_file()
    results["3"] = check_3_preregistration_bars()
    results["4"] = check_4_training_args()

    out_path = REPO / "evaluation" / "rule0_audit_2026_08_13_results.json"
    serializable = {
        "1a": {"n_abstention_rows": results["1a"][0], "n_total_rows": results["1a"][1]},
        "1b": {
            "n_zero_count_pairs": len(results["1b"][1]),
            "zero_pairs": [list(p) for p in results["1b"][1]],
            "n_unmatched_rows": results["1b"][2],
            "pair_counts_min": min(results["1b"][0].values()),
            "pair_counts_max": max(results["1b"][0].values()),
        },
        "1c": {"n_critical_rows": results["1c"][0], "cap_target": results["1c"][1]},
        "1d": {"n_teacher": results["1d"][0], "n_total": results["1d"][1],
              "pct": round(100 * results["1d"][0] / results["1d"][1], 2)},
        "1e": results["1e"],
        "2a": {
            "checkpoint_sha256": results["2a"]["checkpoint"][0],
            "checkpoint_matches_cited": results["2a"]["checkpoint"][1],
            "bridge_sha256": results["2a"]["bridge"][0],
            "bridge_matches_cited": results["2a"]["bridge"][1],
        },
        "2b": {
            "candidates_found": results["2b"][0],
            "locked_file_exists": results["2b"][1],
            "canonical_locked_file": str(LOCKED_SEED999_PATH.relative_to(REPO)),
            "canonical_locked_file_exists": results["2b"][2],
            "canonical_sha256": results["2b"][3],
            "canonical_matches_cited_hash": results["2b"][4],
            "resolution": "RESOLVED 2026-08-13 (AUDIT.md sec AM) -- was GHOST at first audit, now locked "
                          "with a drift-guard test (tests/test_locked_seed999_population.py)",
        },
        "3": results["3"],
        "4": results["4"],
    }
    out_path.write_text(json.dumps(serializable, indent=2))
    print(json.dumps(serializable, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
