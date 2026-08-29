# Vendored STGT front-end (read-only)

Copied from the teammate's retrained-model repo so `stgt_bridge.py` can load
`swarm_data/best_model.pt` and run inference without depending on a second
checked-out repo on disk.

- **Upstream repo**: https://github.com/pizz-beep/capstone
- **Upstream commit**: `b139dcee71feb82244ef1470a6193a628040f318` ("docs: add
  README for LLM handoff", 2026-08-01)
- **Vendored on**: 2026-08-03

## What's here and why

| file | source | notes |
|---|---|---|
| `model.py` | upstream `model.py`, verbatim | `build_graph`, `sequence_to_graphs`, `SpatialGAT`, `PositionalEncoding`, `TemporalTransformer`, `STGTModel`. Architecture must match `best_model.pt`'s `model_state_dict` exactly — do not "clean up" to match this repo's `src/swarm_intent/model.py`, they are similar but not identical (e.g. upstream's `STGTModel.forward` looks up a module-level `device` global; this repo's `model.py` infers device from its own parameters). |
| `config.py` | upstream `config.py`, verbatim | The old-style module-level `CFG` dict / `device` global this repo's `config.py` deliberately replaced (see top-level `CLAUDE.md`: "do not reintroduce module-level globals"). Kept as-is here because `model.py` imports from it — vendored code stays byte-compatible with its own upstream, this package is not asked to conform to this repo's no-globals convention. **Do not import this `config.py` from outside `stgt/`** — use `swarm_intent.config.Config` for anything that isn't purely internal to the vendored model.
| `inference.py` | upstream `inference.py`, **pruned** | Only `predict_v2` and `sliding_window_inference` are copied. |

## Deliberately NOT vendored

Upstream `inference.py` also defines `infer_behavior_trend`, `build_tactical_context`,
and `build_llm_prompt`. **None of these are copied here, on purpose**:

- `build_tactical_context`/`build_llm_prompt` hardcode upstream's own narrative
  vocabulary and `OUTPUT_SCHEMA`, which predate this repo's fixes: `likely_intent`
  is `surveillance / approach / encirclement / retreat / repositioning / unknown`
  (missing every value this repo's evaluator/fine-tuned adapters actually expect —
  see `src/swarm_intent/llm/prompts.py`'s `INTENT_FAMILIES`), and
  `recommended_action` is missing `increase_surveillance`. Vendoring either would
  violate this repo's frozen-vocabulary invariant (`context_spec.py`'s module
  docstring, invariant #1) the moment anything called them. This repo's own
  `src/swarm_intent/inference.py::build_tactical_context`/`build_llm_prompt` (fed by
  `calibration.py` + `context_spec.py`) are the ones actually used — see
  `stgt_bridge.py`.
- Upstream's `build_tactical_context` also emits a literal `⚠` character in one of
  its lines — a character no training prompt in this repo has ever contained.
  Excluding the function sidesteps that too, but `stgt_bridge.py` guards against
  reintroducing it independently (see its own module docstring / tests).
- `infer_behavior_trend` is dead code without `build_tactical_context` (nothing
  else upstream calls it), so it wasn't copied either.

## Usage

This subpackage is a thin, frozen adapter layer. Application code should not call
`predict_v2`/`sliding_window_inference` directly — go through
`src/swarm_intent/stgt_bridge.py`, which converts their output into this repo's
own tactical-context format (`context_spec.py` vocabulary, `calibration.py`
thresholds) rather than upstream's.
