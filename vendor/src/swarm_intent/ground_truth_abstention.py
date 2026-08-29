"""
Phase 3a (docs/PHASE3A_ABSTENTION_CORPUS.md), step 1: the ground-truth abstention-mechanism
classifier. Deliberately independent of src/swarm_intent/stgt_bridge.py and any STGT
checkpoint -- sec AN proved STGT's own per-window formation_history read is noisy enough to
hallucinate a spurious 3rd formation on genuinely 2-hop trajectories (10.2% false-positive
rate when used as a routing signal). Any abstention label derived from that noisy read would
teach a fine-tune to abstain on cases that are actually answerable -- the exact failure this
module exists to avoid.

Ground truth here means the SIMULATOR's own generation parameters: the `chain` a caller
passed to swarm_intent.eval_trajectories.build_long_sequence_labeled (the list of formations
the generator was TOLD to produce, known with certainty because the caller chose it -- not
inferred from anything), and that function's own `true_labels` return value (per-timestep
labels it assigns directly while building the sequence, using TRANSITION_CLASS during each
hop's blend window -- also simulator-authored, never an STGT read).

Validated (docs/PHASE3A_ABSTENTION_CORPUS.md) against AUDIT.md sec AK's 502-case
STGT-derived categorization: 94.6% raw agreement (475/502); 100% of the 27 disagreements
trace to STGT's own read noise (11 cases where STGT failed to detect the multi-hop structure
at all -- the sec AK/AL bucket_A_misrouted and dispersed_converging_ambiguity cases; 16 cases
where STGT's noisy intermediate read flipped the oscillation-vs-multi_hop determination) --
this module was correct in every disagreement, not sec AK's STGT-derived read.
"""
from __future__ import annotations

from .config import TRANSITION_CLASS

MULTI_HOP = "multi_hop"
OSCILLATION = "oscillation"
TERMINAL_TRANSITIONING = "terminal_transitioning"


def classify_trajectory_ground_truth(chain: list[str], true_labels: list[str] | None = None) -> str | None:
    """chain: the TRUE, simulator-known sequence of distinct formations (e.g. the list
    swarm_intent.eval_trajectories.sample_chain produced, or any equivalent caller-chosen
    sequence -- never an STGT read). true_labels: the per-timestep true label array
    build_long_sequence_labeled returns for that chain (optional -- only needed to detect
    terminal_transitioning; multi_hop/oscillation are fully determined by `chain` alone).

    Returns one of MULTI_HOP / OSCILLATION / TERMINAL_TRANSITIONING (a ground-truth
    abstention mechanism), or None if the trajectory is genuinely answerable (chain length
    <=2, and -- if true_labels was supplied -- the observation runs long enough to have
    settled into the final formation's dwell period rather than being cut off mid-blend).

    len(chain)>=3 is sufficient on its own: a trajectory the simulator was told to visit 3+
    distinct formations can never collapse to a single (from, to) RULES pair, regardless of
    how the observation window is framed. terminal_transitioning is the one mechanism that
    depends on true_labels rather than chain alone -- it fires when the observation is
    truncated before the final hop's dwell period, so the destination formation was never
    actually confirmed within the observed window, even though the full chain (if observed
    longer) would resolve to length <=2."""
    if len(chain) >= 3:
        return OSCILLATION if chain[0] == chain[-1] else MULTI_HOP
    if true_labels and true_labels[-1] == TRANSITION_CLASS:
        return TERMINAL_TRANSITIONING
    return None
