"""ROS-state -> LLM tactical-context (`ctx`) builder. Pure Python: NO ROS, NO model.

Isolated on purpose (per integration plan): this is the piece most likely to
need iteration once real STGT `/formation` data flows, and it must be tunable
without touching the node's async/threading plumbing.

It reproduces the OUTPUT FORMAT of aadhya/uav-llm (read-only, pinned commit
69ac2b6) `src/swarm_intent/inference.py:build_tactical_context`, and references
the FROZEN vocabulary CONSTANTS from `src/swarm_intent/context_spec.py` directly
(imported, not copied) so it cannot silently drift from her schema. context_spec
is stdlib-only (torch-free), so importing it pulls in no GPU deps.

DEGENERATE-CTX CONTRACT (v1, until STGT `/formation` is live):
  The `ctx` fields are STGT-derived tactical semantics (formation, velocity /
  stability / spread trends, role differentiation, formation-classifier
  confidence). None of that is expressible from `/tracks` (positions) +
  `/fingerprints` (identity). So with no `/formation` publisher, those fields
  fall back to the frozen NEUTRAL defaults (steady / holding / stable spread /
  not prominent / no transitions) and `formation_present=False`.

  This "no real signal" state is surfaced HONESTLY via the returned
  `formation_present` flag (the node maps it to
  template_id="ctx_degenerate_no_formation") — it is NOT disguised as real
  signal, and it flips automatically to the real path the moment `ctx_builder`
  starts receiving formation data.

  Deliberate non-conflation: the ctx "Classifier confidence" line is STGT
  *formation-classifier* confidence, a DIFFERENT quantity from RF-fingerprint
  *identity* confidence. We do NOT borrow fingerprint confidence into it; with
  no STGT it is honestly reported as 0.00 / all-windows-low.
"""
from swarm_intent import context_spec as spec

DEGENERATE_TEMPLATE_ID = "ctx_degenerate_no_formation"
FORMATION_TEMPLATE_ID = "ctx_formation_stgt"   # real STGT /formation path
UNKNOWN_FORMATION = "unknown"     # honest placeholder: no formation classifier yet
WINDOW_STEP = 50                  # matches build_tactical_context's fixed step label


def _build_real_ctx(snapshot, formation):
    """Real STGT /formation path (lands with the capstone_formation publisher).

    Builds the tactical-context from the classifier's FormationEstimate: real
    formation CLASS + confidence drive the `Dominant formation` / `Confidence`
    lines and the transition line (from->to on a class change). Motion dynamics
    (velocity / stability / spread trend) are held at the NEUTRAL defaults ON
    PURPOSE: the STGT model is translation-invariant (centroid-relative features),
    so it does not track translational velocity — reporting a fabricated velocity
    trend here would be dishonest. So this ctx is "real formation identity, no
    dynamics-trend claim", not a half-real signal disguised as full STGT output.
    """
    t0 = float(snapshot.get("t_start_s", 0.0))
    t1 = float(snapshot.get("t_end_s", t0))
    n = 1
    fclass = str(formation.get("formation_class", UNKNOWN_FORMATION))
    conf = float(formation.get("confidence", 0.0))
    frm, to = formation.get("from"), formation.get("to")

    lines = [
        f"Observation window: {t0:g}s - {t1:g}s ({n} overlapping {WINDOW_STEP}-step windows)",
        f"Dominant formation: {fclass}",
        f"Formation history: {fclass}",
    ]
    if frm and to and frm != to:
        lines.append(f"Transition detected at t={t1:g}s: {frm} -> {to}")
    else:
        lines.append(spec.NO_TRANSITIONS_DETECTED)
    low = 0 if conf >= 0.6 else n
    lines += [
        f"Velocity trend: {spec.VELOCITY_STEADY} (delta_v={0.0:+.2f})",
        f"Formation stability: {spec.STABILITY_HOLDING} (mean={0.0:.2f})",
        f"Spread dynamics: {spec.SPREAD_STABLE} (mean approach_rate={0.0:.3f})",
        f"Role differentiation: {spec.ROLE_DIFFERENTIATION_NOT_PROMINENT}",
        spec.CONFIDENCE_LINE_TEMPLATE.format(mean_conf=conf, low_conf=low),
    ]
    ctx_str = "\n".join(lines)

    key_windows = [{
        "t": f"{t0:g}-{t1:g}s", "formation": fclass, "confidence": conf,
        "velocity": 0.0, "approach": 0.0, "stability": 0.0,
        "from": frm, "to": to,
    }]
    meta = {
        "formation_present": True,
        "template_id": FORMATION_TEMPLATE_ID,
        "k_estimated": int(snapshot.get("k_estimated", 0)),
        "n_tracks": len(snapshot.get("tracks", []) or []),
        "formation_class": fclass,
        "formation_confidence": conf,
    }
    return ctx_str, key_windows, meta


def build_ctx(snapshot):
    """Build the LLM tactical-context from a ROS-state snapshot.

    snapshot keys:
        t_start_s, t_end_s : float  observation window (real sim seconds)
        tracks             : list of dicts (at least {'global_id'})  -- count used
        k_estimated        : int
        formation          : None (no STGT yet) | dict (future enrichment seam)

    Returns (ctx_str, key_windows, meta).
    """
    formation = snapshot.get("formation")
    formation_present = formation is not None

    if formation_present:
        return _build_real_ctx(snapshot, formation)

    # ---------------- DEGENERATE PATH (no STGT / no /formation) ----------------
    t0 = float(snapshot.get("t_start_s", 0.0))
    t1 = float(snapshot.get("t_end_s", t0))
    n = 1  # single snapshot; STGT would provide N overlapping temporal windows

    lines = [
        f"Observation window: {t0:g}s - {t1:g}s ({n} overlapping {WINDOW_STEP}-step windows)",
        f"Dominant formation: {UNKNOWN_FORMATION}",
        f"Formation history: {UNKNOWN_FORMATION}",
        spec.NO_TRANSITIONS_DETECTED,
        f"Velocity trend: {spec.VELOCITY_STEADY} (delta_v={0.0:+.2f})",
        f"Formation stability: {spec.STABILITY_HOLDING} (mean={0.0:.2f})",
        f"Spread dynamics: {spec.SPREAD_STABLE} (mean approach_rate={0.0:.3f})",
        f"Role differentiation: {spec.ROLE_DIFFERENTIATION_NOT_PROMINENT}",
        spec.CONFIDENCE_LINE_TEMPLATE.format(mean_conf=0.0, low_conf=n),
    ]
    ctx_str = "\n".join(lines)

    key_windows = [{
        "t": f"{t0:g}-{t1:g}s",
        "formation": UNKNOWN_FORMATION,
        "confidence": 0.0,
        "velocity": 0.0,
        "approach": 0.0,
        "stability": 0.0,
        "from": None,
        "to": None,
    }]

    meta = {
        "formation_present": False,
        "template_id": DEGENERATE_TEMPLATE_ID,
        "k_estimated": int(snapshot.get("k_estimated", 0)),
        "n_tracks": len(snapshot.get("tracks", []) or []),
    }
    return ctx_str, key_windows, meta
