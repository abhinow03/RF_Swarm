"""
Phase 3a LOCKED CONFIG, mandatory first action: verify checkpoints/v5_sft_v5a_PROTECTED/ is a
byte-identical, read-only copy of checkpoints/v5_sft/ (v5-a's shipped adapter). This is the
live, re-runnable check backing that claim -- not a one-time manual `cp`+`sha256sum` left
undocumented. Run this any time before/after Phase 3a work to prove the safety copy still
matches and was never touched.

Usage:
    python scripts/phase3a_verify_safety_copy.py
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ORIGINAL = REPO / "checkpoints" / "v5_sft"
PROTECTED = REPO / "checkpoints" / "v5_sft_v5a_PROTECTED"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = sha256_file(p)
    return out


def main():
    if not ORIGINAL.is_dir():
        print(f"FAIL: {ORIGINAL} does not exist"); sys.exit(1)
    if not PROTECTED.is_dir():
        print(f"FAIL: {PROTECTED} does not exist -- safety copy missing"); sys.exit(1)

    orig_hashes = hash_tree(ORIGINAL)
    prot_hashes = hash_tree(PROTECTED)

    missing_in_protected = sorted(set(orig_hashes) - set(prot_hashes))
    extra_in_protected = sorted(set(prot_hashes) - set(orig_hashes))
    mismatched = sorted(f for f in (set(orig_hashes) & set(prot_hashes))
                        if orig_hashes[f] != prot_hashes[f])

    # read-only check: no file under PROTECTED may have a write bit set for anyone
    writable_files = []
    for p in PROTECTED.rglob("*"):
        mode = p.stat().st_mode
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            writable_files.append(str(p.relative_to(PROTECTED)))

    ok = (not missing_in_protected and not extra_in_protected and not mismatched
         and not writable_files)

    result = {
        "n_files_original": len(orig_hashes), "n_files_protected": len(prot_hashes),
        "missing_in_protected": missing_in_protected, "extra_in_protected": extra_in_protected,
        "mismatched_hashes": mismatched, "writable_paths_found": writable_files,
        "all_hashes_match": not mismatched and not missing_in_protected and not extra_in_protected,
        "read_only_enforced": not writable_files,
        "overall_ok": ok,
        "original_hashes": orig_hashes, "protected_hashes": prot_hashes,
    }
    out_path = REPO / "evaluation" / "phase3a_safety_copy_verification.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"original files: {len(orig_hashes)}  protected files: {len(prot_hashes)}")
    print(f"missing_in_protected: {missing_in_protected}")
    print(f"extra_in_protected: {extra_in_protected}")
    print(f"mismatched_hashes: {mismatched}")
    print(f"writable_paths_found (should be empty): {writable_files}")
    print(f"\nOVERALL: {'PASS -- safety copy verified byte-identical and read-only' if ok else 'FAIL'}")
    print(f"saved {out_path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
