"""Offline testing: before an LLM-generated candidate is ever run for real
against the environment, cheaply test it against the parent node's own
already-collected transitions (no ``env.step()`` calls, no evaluation
budget spent) and decide whether it's worth promoting to a real Node at
all.

Strategy ``"behavioral_similarity"``: generate K independent candidates
from the same edge/parent (see ``core.training.generate_candidate_node``),
score each by how well its proposed actions agree with what the parent
actually did -- weighted by how good the parent's action actually was at
each transition (its *advantage*: this iteration's own preprocessed
return, centered by the mean over that same transition set, or raw reward
when preprocessing is ``"raw"`` -- see ``core.evidence_preprocessing``).
Agreeing with the parent on a transition where it did well is rewarded;
agreeing on a transition where it did poorly is penalized, and vice versa
for disagreeing -- so a candidate is naturally pushed to imitate a
successful parent and diverge from a struggling one, without needing any
separate "is this node good" check.

A candidate whose code doesn't even compile/define ``policy`` automatically
loses (its score is ``None`` -- never comparable to a real score, never a
winner). A candidate that raises or proposes an invalid action on some
specific transition is scored as the worst possible outcome for *that*
transition (more strongly penalized than merely disagreeing), not skipped
or treated as neutral.

The winner (if any candidate clears ``acceptance_threshold``) is the only
one a caller should ever promote to a real Node -- the other K-1 (and any
that never even validated) are discarded after scoring and never persisted
as Nodes at all (see ``core.edges.generate_edge_output``/``materialize_node``).
When no candidate passes, the caller falls back to *reevaluating the
parent* for real instead of treating this as a failure (see
``core.training.run_training_loop`` / ``core.mcts.run_mcts_search``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.evidence_preprocessing import ProcessedTransition
from execution.policy_runner import PolicyRunner

NONE_STRATEGY = "none"
BEHAVIORAL_SIMILARITY_STRATEGY = "behavioral_similarity"
VALID_OFFLINE_TEST_STRATEGIES = (NONE_STRATEGY, BEHAVIORAL_SIMILARITY_STRATEGY)

# For now K=1/threshold=0.5 are simply reasonable starting points -- with
# strategy defaulting to "none", neither matters until a researcher opts in.
DEFAULT_K = 1
DEFAULT_ACCEPTANCE_THRESHOLD = 0.5


@dataclass
class OfflineTestConfig:
    """``strategy="none"`` (the default) skips offline testing entirely --
    every existing Train config/behavior is unaffected, and this is never
    applied to the very first (root) node regardless of ``strategy``,
    since there's no parent trajectory yet to test against. ``k`` is how
    many independent candidates to generate and offline-compare per
    iteration; ``acceptance_threshold`` is the minimum normalized score
    (see :func:`score_candidate`, roughly in ``[-1, 1]`` except when a
    candidate errors on some transitions, which can push it lower) a
    candidate must clear to be promoted at all."""

    strategy: str = NONE_STRATEGY
    k: int = DEFAULT_K
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD

    def __post_init__(self):
        if self.strategy not in VALID_OFFLINE_TEST_STRATEGIES:
            raise ValueError(f"Unknown offline_test strategy: {self.strategy!r} "
                              f"(choices: {VALID_OFFLINE_TEST_STRATEGIES})")
        if self.k <= 0:
            raise ValueError("k must be positive.")


@dataclass
class CandidateScore:
    index: int
    score: Optional[float]  # None only if the code never even validated/ran


@dataclass
class OfflineTestResult:
    passed: bool
    winner_index: Optional[int]
    scores: list[CandidateScore] = field(default_factory=list)


def _transition_value(processed: ProcessedTransition) -> float:
    """This transition's 'how good was the parent's action here' signal --
    its preprocessed return when available, else its own immediate reward
    (covers ``raw`` mode, where ``return_value`` is always ``None``, and
    the rarer episodic/k-step 'unavailable for this transition' case)."""
    if processed.return_value is not None:
        return processed.return_value
    return processed.transition.reward


def score_candidate(context, processed_transitions: list[ProcessedTransition],
                     candidate_code: Optional[str], step_timeout: float = 2.0) -> Optional[float]:
    """Runs ``candidate_code`` offline -- never against the live
    environment, one action proposal per already-recorded transition, via
    the same sandboxed :class:`~execution.policy_runner.PolicyRunner`
    Play/Runs already use -- and returns its normalized behavioral-
    similarity score, or ``None`` if the code itself never even
    compiles/defines ``policy`` (an automatic loss, never comparable to a
    real score). See module docstring for the scoring rule."""
    if not candidate_code or not processed_transitions:
        return None

    runner = PolicyRunner(candidate_code, step_timeout=step_timeout)
    if not runner.ready:
        runner.close()
        return None

    try:
        values = [_transition_value(p) for p in processed_transitions]
        mean_value = sum(values) / len(values)
        advantages = [v - mean_value for v in values]
        mean_abs_advantage = sum(abs(a) for a in advantages) / len(advantages)
        if mean_abs_advantage == 0:
            # Every transition was equally good/bad -- no signal to select
            # a winner on, but the code itself is valid, so this isn't a
            # loss either.
            return 0.0
        max_abs_advantage = max(abs(a) for a in advantages)

        total = 0.0
        for processed, advantage in zip(processed_transitions, advantages):
            transition = processed.transition
            observation = context.experience.read_state(transition, which="state")
            # Each proposal is scored against an arbitrary, possibly
            # out-of-episode-order historical transition -- not a live
            # trajectory -- so there's no real "previous step" to carry
            # memory from; a fresh {} every call means a memory-aware
            # candidate is never offline-tested on its actual memory use,
            # only on its per-observation action choice (same limitation as
            # any other offline/off-policy proposal check).
            outcome = runner.act(observation, {})
            if outcome.ok and context.adapter.is_valid_action(outcome.action):
                proposed = context.adapter.normalize_action(outcome.action)
                agree = proposed == transition.action
                total += advantage if agree else -advantage
            else:
                # Couldn't even propose a usable action here -- worse than
                # simply disagreeing (see module docstring).
                total += -max_abs_advantage
        return (total / len(processed_transitions)) / mean_abs_advantage
    finally:
        runner.close()


def run_offline_test(context, processed_transitions: list[ProcessedTransition],
                      candidate_codes: list[Optional[str]], config: OfflineTestConfig,
                      step_timeout: float = 2.0) -> OfflineTestResult:
    """Scores every one of ``candidate_codes`` (already independently
    generated -- see ``core.training.generate_candidate_node``) and picks
    the best one, if any clears ``config.acceptance_threshold``."""
    scores: list[CandidateScore] = []
    best_index: Optional[int] = None
    best_score: Optional[float] = None
    for i, code in enumerate(candidate_codes):
        score = score_candidate(context, processed_transitions, code, step_timeout=step_timeout)
        scores.append(CandidateScore(index=i, score=score))
        if score is not None and score > config.acceptance_threshold:
            if best_score is None or score > best_score:
                best_score = score
                best_index = i
    return OfflineTestResult(passed=best_index is not None, winner_index=best_index, scores=scores)
