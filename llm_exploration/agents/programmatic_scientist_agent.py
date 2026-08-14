"""ProgrammaticScientistAgent: a modular architecture for learning executable
programmatic policies from environment interaction.

This agent treats policy learning as a small research loop: it keeps what
actually happened (experience) separate from what it believes about the
environment (belief), separate from uncertain claims worth testing
(hypotheses), separate from the executable candidates it is trying
(programs), and separate from the logic that decides which candidate to run
or improve next (search). Each of these concerns lives in its own small
class so any one of them can be swapped out later (e.g. replacing the search
controller with MCTS) without touching the rest of the agent.

# ---------------------------------------------------------------------------
# Conceptual separation
# ---------------------------------------------------------------------------
#
# Experience:
#     Objective records of what happened: (s, a, r, s').
#
# Environment belief:
#     The agent's current interpretation of environment dynamics/rewards.
#
# Hypothesis:
#     An uncertain, falsifiable claim that may motivate future exploration.
#
# Program:
#     An executable policy candidate.
#
# Search controller:
#     Allocates computation/environment interaction across program candidates.
#
# These concepts intentionally remain separate. A program may reference
# multiple hypotheses, and one hypothesis may motivate multiple programs.
# ---------------------------------------------------------------------------

This is a deliberately simple first implementation. See the docstring/comments
on each component for the more sophisticated variants they could grow into.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from llm import LLMClient, ChatSession

from .base import BaseAgent


# ============================================================================
# Prompts
# ============================================================================

PROGRAM_GENERATION_SYSTEM_PROMPT = """You are writing an executable Python policy for a \
reinforcement learning environment.

Observation space: {observation_space_description}
Action space: {action_space_description}

You do not choose actions directly. Instead you write a Python program that defines:
    def policy(observation):
        ...
        return action
This function is called once per environment step and must return a valid action."""

PROGRAM_GENERATION_PROMPT_TEMPLATE = """Write an improved policy program for this environment.

Reason for this generation: {generation_reason}

Parent policy (or "None - write from scratch" if there is no parent):
{parent_program}

Environment beliefs (may be empty):
{environment_belief}

Active hypotheses (may be empty):
{hypotheses}

Recent interaction evidence:
{recent_transitions}

Requirements:
- Return ONLY raw Python source code. No Markdown, no code fences, no commentary.
- Define exactly one required entry point:
    def policy(observation):
        ...
        return action
- The returned action must be valid for the action space described above.
- Do not use "import" statements at all. `np` (numpy) and `math` are already available \
as globals, ready to use directly (e.g. `np.array(...)`).
- Use the parent policy, evidence, beliefs, and hypotheses above to write a policy that \
improves on the parent (or write a reasonable first policy if there is no parent).

Respond with only the Python source code."""

RETRY_FEEDBACK_TEMPLATE = """

Your previous candidate could not be used ({error}). Respond again with a corrected, \
complete program. Remember: raw Python source only, no Markdown/commentary, and it must \
define `def policy(observation): ... return action`."""

ENVIRONMENT_UNDERSTANDING_SYSTEM_PROMPT = """You are a careful scientist studying an \
unfamiliar reinforcement learning environment through direct observation of \
(state, action, reward, next_state) transitions.

Observation space: {observation_space_description}
Action space: {action_space_description}

You maintain a short, structured belief about the environment's dynamics and reward \
rules. Clearly distinguish facts you have directly observed evidence for from things \
that remain uncertain or are only guessed. Do not overclaim a rule from a single \
transition."""

ENVIRONMENT_UNDERSTANDING_PROMPT_TEMPLATE = """Current belief (or empty if none yet):
{current_belief}

New transitions observed since the last update:
{recent_transitions}

Update the belief. Respond in exactly this format (each section a short bullet list, \
use "- none yet" if a section is empty):

KNOWN_RULES:
- ...

UNCERTAIN_RULES:
- ...

REWARD_RULES:
- ...

STRATEGY_NOTES:
- ...

Preserve useful previous beliefs; revise anything contradicted by the new transitions. \
Be concise."""


# ============================================================================
# Restricted program execution
# ============================================================================

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "map": map, "filter": filter, "isinstance": isinstance,
    "True": True, "False": False, "None": None,
}


def _compile_policy_source(source: str) -> tuple[Callable[[Any], Any] | None, str | None]:
    """Compile ``source`` under a restricted globals dict.

    Rejects import statements via AST inspection (rather than trusting the
    prompt), then executes the module body and pulls out ``policy``.

    Returns:
        ``(policy_fn, error)``: exactly one of the two is ``None``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return None, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return None, "Imports are not allowed in generated programs."

    restricted_globals: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "np": np,
        "math": math,
    }
    try:
        exec(compile(tree, "<candidate_policy>", "exec"), restricted_globals)
    except Exception as e:
        return None, f"Compilation/execution failed: {e}"

    policy_fn = restricted_globals.get("policy")
    if not callable(policy_fn):
        return None, "Program does not define a callable policy(observation) function."
    return policy_fn, None


def _strip_code_fences(text: str) -> str:
    """Strip an accidental single Markdown code fence wrapping the whole response."""
    text = text.strip()
    fence_match = re.match(r"^```[ \t]*\w*[ \t]*\r?\n(.*)\r?\n```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return text


def is_valid_action(action_space, action: Any) -> bool:
    """Validate an action against ``action_space``, with light normalization
    (e.g. accepting numpy ints / bools for a Discrete space)."""
    try:
        if action_space.contains(action):
            return True
    except Exception:
        pass

    if hasattr(action_space, "n"):
        try:
            normalized = int(action)
        except (TypeError, ValueError):
            return False
        try:
            return action_space.contains(normalized)
        except Exception:
            return 0 <= normalized < action_space.n

    return False


def normalize_action(action_space, action: Any) -> Any:
    """Coerce an already-valid action into the canonical type the env expects."""
    if hasattr(action_space, "n"):
        try:
            return int(action)
        except (TypeError, ValueError):
            return action
    return action


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Transition:
    """One objective record of what happened: (s, a, r, s').

    ``proposed_action`` is what the active program's ``policy()`` returned;
    ``executed_action`` is what was actually sent to the environment. They
    differ whenever the proposed action was invalid/erroring and a random
    fallback action was executed instead - keeping them separate means a
    fallback's poor outcome is never credited (or blamed) on the program.
    """

    observation: Any
    proposed_action: Any
    executed_action: Any
    reward: float
    next_observation: Any
    done: bool
    program_id: int | None
    rollout_id: int | None
    episode_id: int
    execution_error: str | None = None
    id: int = -1


@dataclass
class Rollout:
    """A contiguous stretch of transitions executed under one program.

    Usually one rollout == one episode, but a rollout ends early whenever the
    search controller swaps in a different active program mid-episode, so a
    rollout should be read as "one continuous run of a specific program",
    not "one episode".
    """

    id: int
    program_id: int | None
    transitions: list[Transition] = field(default_factory=list)
    total_return: float = 0.0
    completed: bool = False


@dataclass
class ProgramStatistics:
    """Running statistics for one program.

    ``total_reward``/``num_rollouts`` are updated only at rollout completion
    (from ``Rollout.total_return``), so ``mean_return`` is a true mean
    rollout return - not total reward divided by step count, which would
    silently conflate "many short bad rollouts" with "few long good ones".
    ``num_steps``/``execution_errors`` are updated per-transition instead,
    since they describe per-step reliability rather than per-rollout return.
    """

    num_steps: int = 0
    num_rollouts: int = 0
    total_reward: float = 0.0
    execution_errors: int = 0

    @property
    def mean_return(self) -> float:
        if self.num_rollouts == 0:
            return 0.0
        return self.total_reward / self.num_rollouts


@dataclass
class ProgramCandidate:
    """A candidate executable policy (metadata only; see ``ProgramStore`` for
    the compiled callable itself, which is intentionally not stored here so
    this dataclass stays easy to log/serialize)."""

    id: int
    source: str
    parent_id: int | None = None
    generation_reason: str | None = None
    hypothesis_ids: list[int] = field(default_factory=list)
    statistics: ProgramStatistics = field(default_factory=ProgramStatistics)
    validation_error: str | None = None
    is_valid: bool = False
    rollout_ids: list[int] = field(default_factory=list)


@dataclass
class ValidationResult:
    valid: bool
    error: str | None
    policy_fn: Callable[[Any], Any] | None


@dataclass
class EnvironmentBelief:
    """The agent's current (deliberately simple/textual) interpretation of
    the environment's dynamics and rewards."""

    known_rules: list[str] = field(default_factory=list)
    uncertain_rules: list[str] = field(default_factory=list)
    reward_rules: list[str] = field(default_factory=list)
    useful_strategy_notes: list[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    """An uncertain, falsifiable claim - NOT a program.

    e.g. Hypothesis: "Goal Y only gives reward after activating X."
         Program:    an executable policy designed to exploit or test that.

    One hypothesis may motivate many programs, and one program may reference
    many hypotheses (via ``ProgramCandidate.hypothesis_ids``); the two are
    intentionally never merged into a single identity.
    """

    id: int
    claim: str
    confidence: float = 0.5
    supporting_transition_ids: list[int] = field(default_factory=list)
    contradicting_transition_ids: list[int] = field(default_factory=list)
    status: str = "untested"


@dataclass
class ExperimentPlan:
    """hypothesis + current knowledge -> a desired experiment/behavior to run.

    e.g. Hypothesis: "X must be activated before Y."
         Experiment: "Navigate directly to Y without touching X."
         The ProgramGenerator then synthesizes a policy performing that experiment.

    Not used by the baseline agent yet; see ``ExperimentPlanner``.
    """

    hypothesis_id: int | None
    description: str


@dataclass
class ProgramGenerationContext:
    """Explicit context handed to the generator, instead of it reaching into
    the agent for whatever it wants."""

    parent_program: ProgramCandidate | None
    recent_transitions: list[Transition]
    environment_belief: EnvironmentBelief | None
    hypotheses: list[Hypothesis]
    generation_reason: str | None = None
    # Set by the agent's retry loop with the previous attempt's validation
    # error, so the LLM sees concrete feedback instead of repeating itself.
    retry_feedback: str | None = None


# ============================================================================
# Formatting helpers
# ============================================================================
#
# Kept centralized so no component re-implements its own observation/action
# space description or transition formatting. Environment-specific encoders
# (e.g. for image observations) could replace ``format_observation`` later.
# ============================================================================

def format_observation(obs: Any) -> str:
    """Format an observation for a prompt, truncating large arrays/collections."""
    if isinstance(obs, np.ndarray):
        flat = obs.flatten()
        if flat.size > 20:
            return f"{list(flat[:20])}... (truncated, full shape={obs.shape})"
        return str(obs.tolist())
    if isinstance(obs, (list, tuple)):
        if len(obs) > 10:
            return f"{list(obs[:10])}... (truncated, {len(obs)} items total)"
        return str(list(obs))
    if isinstance(obs, dict):
        items = list(obs.items())
        if len(items) > 5:
            return f"{dict(items[:5])}... (truncated, {len(items)} keys total)"
        return str(dict(items))
    return str(obs)


def describe_observation_space(space: Any) -> str:
    """Describe an observation space for the LLM."""
    if space is None:
        return "Unknown (not provided); infer structure from the observations shown below."
    try:
        if hasattr(space, "shape") and hasattr(space, "low"):
            return f"Array of shape {space.shape} with values in range [{space.low}, {space.high}]"
        if hasattr(space, "n"):
            return f"Discrete: integers from 0 to {space.n - 1}"
        if hasattr(space, "spaces") and isinstance(getattr(space, "spaces"), dict):
            fields_desc = ", ".join(space.spaces.keys())
            return f"Dict with fields: {fields_desc}"
        return str(space)
    except Exception:
        return str(space)


def describe_action_space(action_space: Any) -> str:
    """Describe an action space for the LLM."""
    try:
        if hasattr(action_space, "n"):
            return f"Discrete actions: integers from 0 to {action_space.n - 1}"
        if hasattr(action_space, "shape") and hasattr(action_space, "low"):
            return (f"Continuous actions: array of shape {action_space.shape} with values "
                    f"in range [{action_space.low}, {action_space.high}]")
        return str(action_space)
    except Exception:
        return str(action_space)


def format_transition(t: Transition) -> str:
    line = (f"  step {t.id}: obs={format_observation(t.observation)} "
            f"proposed_action={t.proposed_action} executed_action={t.executed_action} "
            f"reward={t.reward:.3f} done={t.done}")
    if t.execution_error:
        line += f" error={t.execution_error}"
    return line


def format_transitions(transitions: list[Transition], max_shown: int = 20) -> str:
    if not transitions:
        return "No transitions recorded yet."
    shown = transitions[-max_shown:]
    prefix = ""
    if len(transitions) > len(shown):
        prefix = f"[showing last {len(shown)} of {len(transitions)} transitions]\n"
    return prefix + "\n".join(format_transition(t) for t in shown)


def format_environment_belief(belief: EnvironmentBelief) -> str:
    def _section(title: str, items: list[str]) -> str:
        if not items:
            return f"{title}: none yet."
        bullets = "\n".join(f"  - {item}" for item in items)
        return f"{title}:\n{bullets}"

    return "\n".join([
        _section("Known rules", belief.known_rules),
        _section("Uncertain rules", belief.uncertain_rules),
        _section("Reward rules", belief.reward_rules),
        _section("Strategy notes", belief.useful_strategy_notes),
    ])


def format_hypotheses(hypotheses: list[Hypothesis]) -> str:
    if not hypotheses:
        return "No hypotheses recorded."
    return "\n".join(
        f"H{h.id} [{h.status}, confidence={h.confidence:.2f}]: {h.claim}"
        for h in hypotheses
    )


def format_program_search(programs: list[ProgramCandidate]) -> str:
    if not programs:
        return "No programs generated yet."
    lines = []
    for p in sorted(programs, key=lambda program: program.id):
        stats = p.statistics
        line = (f"P{p.id} (parent={p.parent_id}, valid={p.is_valid}, "
                f"reason={p.generation_reason!r}): rollouts={stats.num_rollouts} "
                f"mean_return={stats.mean_return:.2f} execution_errors={stats.execution_errors}")
        if p.validation_error:
            line += f" validation_error={p.validation_error!r}"
        lines.append(line)
    return "\n".join(lines)


# ============================================================================
# Experience Store
# ============================================================================

class ExperienceStore:
    """Stores and retrieves objective interaction data only.

    No LLM reasoning and no program-selection logic lives here - this is
    purely an experience database for this agent, not a neural-network
    replay buffer.

    Future sampling strategies this could grow (as new query methods):
        - recent (implemented)
        - program-specific (implemented)
        - failure-focused
        - high-reward / reward-prioritized
        - novelty-based
        - contrastive success/failure examples
        - diverse-state sampling
    """

    def __init__(self) -> None:
        self.transitions: list[Transition] = []
        self.rollouts: dict[int, Rollout] = {}
        self._next_transition_id = 0
        self._next_rollout_id = 0

    def start_rollout(self, program_id: int | None) -> int:
        rollout_id = self._next_rollout_id
        self._next_rollout_id += 1
        self.rollouts[rollout_id] = Rollout(id=rollout_id, program_id=program_id)
        return rollout_id

    def finish_rollout(self, rollout_id: int) -> Rollout | None:
        rollout = self.rollouts.get(rollout_id)
        if rollout is None:
            return None
        rollout.completed = True
        return rollout

    def add_transition(
        self,
        rollout_id: int | None,
        observation: Any,
        proposed_action: Any,
        executed_action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
        program_id: int | None,
        episode_id: int,
        execution_error: str | None = None,
    ) -> Transition:
        transition = Transition(
            observation=observation,
            proposed_action=proposed_action,
            executed_action=executed_action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            program_id=program_id,
            rollout_id=rollout_id,
            episode_id=episode_id,
            execution_error=execution_error,
            id=self._next_transition_id,
        )
        self._next_transition_id += 1
        self.transitions.append(transition)

        rollout = self.rollouts.get(rollout_id) if rollout_id is not None else None
        if rollout is not None:
            rollout.transitions.append(transition)
            rollout.total_return += reward

        return transition

    def get_recent_transitions(self, n: int) -> list[Transition]:
        if n <= 0:
            return []
        return self.transitions[-n:]

    def get_transitions_for_program(self, program_id: int) -> list[Transition]:
        return [t for t in self.transitions if t.program_id == program_id]

    def get_rollouts_for_program(self, program_id: int) -> list[Rollout]:
        return [r for r in self.rollouts.values() if r.program_id == program_id]

    def get_successful_rollouts(self, min_return: float = 0.0) -> list[Rollout]:
        return [r for r in self.rollouts.values() if r.completed and r.total_return > min_return]

    def get_failure_rollouts(self, max_return: float = 0.0) -> list[Rollout]:
        return [r for r in self.rollouts.values() if r.completed and r.total_return <= max_return]


# ============================================================================
# Program Store
# ============================================================================

class ProgramStore:
    """Owns candidate programs and their compiled callables.

    Compiled policy functions are kept separately from ``ProgramCandidate``
    (which stays a plain, loggable dataclass) so serializing/inspecting
    program metadata never has to worry about non-serializable callables.

    Parent links (``ProgramCandidate.parent_id``) let the program search
    graph be reconstructed later without forcing a strict tree - a program
    could in principle be reachable from more than one parent-search path.
    """

    def __init__(self) -> None:
        self.programs: dict[int, ProgramCandidate] = {}
        self.compiled_policies: dict[int, Callable[[Any], Any]] = {}
        self._next_program_id = 0

    def add_program(
        self,
        source: str,
        parent_id: int | None = None,
        generation_reason: str | None = None,
        hypothesis_ids: list[int] | None = None,
    ) -> ProgramCandidate:
        program_id = self._next_program_id
        self._next_program_id += 1
        candidate = ProgramCandidate(
            id=program_id,
            source=source,
            parent_id=parent_id,
            generation_reason=generation_reason,
            hypothesis_ids=list(hypothesis_ids or []),
        )
        self.programs[program_id] = candidate
        return candidate

    def get_program(self, program_id: int) -> ProgramCandidate | None:
        return self.programs.get(program_id)

    def get_policy(self, program_id: int) -> Callable[[Any], Any] | None:
        return self.compiled_policies.get(program_id)

    def set_compiled_policy(self, program_id: int, fn: Callable[[Any], Any]) -> None:
        self.compiled_policies[program_id] = fn

    def mark_valid(self, program_id: int, policy_fn: Callable[[Any], Any]) -> None:
        program = self.programs[program_id]
        program.is_valid = True
        program.validation_error = None
        self.set_compiled_policy(program_id, policy_fn)

    def mark_invalid(self, program_id: int, error: str) -> None:
        program = self.programs[program_id]
        program.is_valid = False
        program.validation_error = error

    def record_rollout(self, program_id: int, rollout_id: int) -> None:
        program = self.programs.get(program_id)
        if program is not None:
            program.rollout_ids.append(rollout_id)

    def get_all_programs(self) -> list[ProgramCandidate]:
        return list(self.programs.values())

    def get_valid_programs(self) -> list[ProgramCandidate]:
        return [p for p in self.programs.values() if p.is_valid]


# ============================================================================
# Program Validator
# ============================================================================

class ProgramValidator:
    """Two-stage candidate validation: static, then offline behavioral.

    A failed candidate must NEVER destroy the current working program - this
    class only ever reports pass/fail; the caller decides whether to switch
    the active program (only after ``valid`` is ``True``).

    Implemented now:
        - static: syntax, no imports (AST check), compiles, defines a
          callable ``policy(observation)``, restricted globals.
        - behavioral: replay a handful of real historical observations
          through the candidate and check it doesn't raise or return an
          invalid action.

    Future extensions (not implemented yet):
        - regression tests against known successful states
        - timeout checking
        - counterfactual evaluation
        - world-model simulation
        - formal verification
        - behavioral constraints
    """

    def __init__(self, action_space, num_replay_observations: int = 5) -> None:
        self.action_space = action_space
        self.num_replay_observations = max(0, num_replay_observations)

    def validate(self, source: str, experience_store: ExperienceStore) -> ValidationResult:
        static_result = self._validate_static(source)
        if not static_result.valid:
            return static_result

        error = self._validate_behavioral(static_result.policy_fn, experience_store)
        if error is not None:
            return ValidationResult(valid=False, error=error, policy_fn=None)

        return static_result

    def _validate_static(self, source: str) -> ValidationResult:
        policy_fn, error = _compile_policy_source(source)
        if error is not None:
            return ValidationResult(valid=False, error=error, policy_fn=None)
        return ValidationResult(valid=True, error=None, policy_fn=policy_fn)

    def _validate_behavioral(
        self, policy_fn: Callable[[Any], Any], experience_store: ExperienceStore
    ) -> str | None:
        sample_transitions = experience_store.get_recent_transitions(self.num_replay_observations)
        for transition in sample_transitions:
            try:
                action = policy_fn(transition.observation)
            except Exception as e:
                return f"Raised an exception on a replayed observation: {e}"
            if not is_valid_action(self.action_space, action):
                return f"Returned an invalid action ({action!r}) on a replayed observation."
        return None


# ============================================================================
# Program Generator
# ============================================================================

class ProgramGenerator:
    """Uses the LLM to produce one candidate Python policy from context.

    Implements a single, straightforward strategy: given the parent program
    (if any) plus recent evidence/beliefs/hypotheses, ask the LLM to write
    an improved (or first) policy in one call. This class does not validate
    its own output - that is ``ProgramValidator``'s job.

    Future generator variants (not implemented yet):
        1. Generate from scratch
        2. Rewrite current program (closest to what's implemented here)
        3. Minimal patch
        4. Failure-conditioned generation
        5. Hypothesis-conditioned generation
        6. Generate K diverse programs
        7. Evolutionary mutation/crossover
        8. Hierarchical/sub-policy generation
    """

    def __init__(self, client: LLMClient, observation_space, action_space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        system_prompt = PROGRAM_GENERATION_SYSTEM_PROMPT.format(
            observation_space_description=describe_observation_space(observation_space),
            action_space_description=describe_action_space(action_space),
        )
        # max_messages=1: each generation call is an independent request, not
        # a growing multi-turn conversation (matches the other LLM agents).
        self.chat = ChatSession(client, system=system_prompt, max_messages=1)

    def generate(self, context: ProgramGenerationContext) -> str:
        """Return raw Python source for a candidate ``policy(observation)``.

        Raises on a failed/empty LLM response; the caller is responsible for
        retrying (with updated ``context.retry_feedback``) and for validating
        the returned source.
        """
        prompt = self._build_prompt(context)
        response = self.chat.send(prompt)
        source = _strip_code_fences(response)
        if not source.strip():
            raise ValueError("LLM returned an empty program.")
        return source

    def _build_prompt(self, context: ProgramGenerationContext) -> str:
        parent_text = "None - write from scratch."
        if context.parent_program is not None:
            stats = context.parent_program.statistics
            parent_text = (
                f"{context.parent_program.source}\n\n"
                f"(parent statistics: {stats.num_rollouts} rollout(s), "
                f"mean_return={stats.mean_return:.2f}, "
                f"execution_errors={stats.execution_errors})"
            )

        belief_text = "No environment beliefs recorded yet."
        if context.environment_belief is not None:
            belief_text = format_environment_belief(context.environment_belief)

        hypotheses_text = "No active hypotheses."
        if context.hypotheses:
            hypotheses_text = "\n".join(
                f"- {h.claim} (confidence={h.confidence:.2f})" for h in context.hypotheses
            )

        prompt = PROGRAM_GENERATION_PROMPT_TEMPLATE.format(
            generation_reason=context.generation_reason or "improve the current policy",
            parent_program=parent_text,
            environment_belief=belief_text,
            hypotheses=hypotheses_text,
            recent_transitions=format_transitions(context.recent_transitions),
        )
        if context.retry_feedback:
            prompt += RETRY_FEEDBACK_TEMPLATE.format(error=context.retry_feedback)
        return prompt


# ============================================================================
# Program Evaluator
# ============================================================================

class ProgramEvaluator:
    """Experience -> estimated quality of a program.

    Kept intentionally simple: score is mean rollout return with a small
    penalty proportional to the per-step execution error rate. Raw
    statistics (``ProgramStatistics``) are updated here but kept separate
    from the derived search score, so future scoring strategies can be
    swapped in without touching how statistics are collected.

    Future scoring variants (not implemented yet):
        - success rate
        - return variance
        - uncertainty bonuses
        - risk-sensitive score
        - novelty
        - coverage
        - robustness
        - program complexity penalty
        - multi-objective evaluation
    """

    def __init__(self, error_penalty: float = 1.0) -> None:
        self.error_penalty = error_penalty

    def update_from_transition(
        self, program_id: int, program_store: ProgramStore, transition: Transition
    ) -> None:
        program = program_store.get_program(program_id)
        if program is None:
            return
        stats = program.statistics
        stats.num_steps += 1
        if transition.execution_error is not None:
            stats.execution_errors += 1

    def update_from_rollout(
        self, program_id: int, program_store: ProgramStore, rollout: Rollout
    ) -> None:
        program = program_store.get_program(program_id)
        if program is None:
            return
        stats = program.statistics
        stats.num_rollouts += 1
        stats.total_reward += rollout.total_return

    def score(self, program_id: int, program_store: ProgramStore) -> float:
        program = program_store.get_program(program_id)
        if program is None:
            return float("-inf")
        stats = program.statistics
        if stats.num_rollouts == 0:
            # Not evaluated yet - neutral rather than automatically best/worst.
            return 0.0
        error_rate = stats.execution_errors / stats.num_steps if stats.num_steps else 0.0
        return stats.mean_return - self.error_penalty * error_rate


# ============================================================================
# Execution Manager
# ============================================================================

class ExecutionManager:
    """Decides how long the active program stays active before the search
    controller is consulted again.

    Implemented now: a fixed step budget. The agent never hard-codes this
    threshold itself - it always asks ``should_stop``.

    Future variants (not implemented yet):
        - fixed number of episodes
        - until terminal state
        - until failure/stuck state
        - confidence-based evaluation
        - sequential statistical testing
        - adaptive racing between programs
    """

    def __init__(self, steps_per_program: int = 20) -> None:
        self.steps_per_program = max(1, steps_per_program)

    def should_stop(
        self, program_id: int | None, steps_used: int, current_rollout: int | None
    ) -> bool:
        return steps_used >= self.steps_per_program


# ============================================================================
# Search Controller
# ============================================================================

class SearchController:
    """Answers two questions: which program should run next, and which
    program should parent the next generated candidate.

    Baseline implemented now: current-best / hill climbing - always pick the
    highest-scoring valid program for both questions. Program parent links
    are enough to support a tree/DAG later; nothing here assumes a strict
    tree.

    Future search strategies (not implemented yet):
        1. Linear iterative improvement (closest to what's implemented here)
        2. Greedy best-first search
        3. Beam search
        4. Evolutionary population search
        5. Multi-armed-bandit program allocation
        6. MCTS / UCB program search
        7. Novelty search
        8. Quality-diversity search
    """

    def __init__(self, evaluator: ProgramEvaluator) -> None:
        self.evaluator = evaluator

    def select_program_to_execute(self, program_store: ProgramStore) -> int | None:
        valid_programs = program_store.get_valid_programs()
        if not valid_programs:
            return None
        best = max(valid_programs, key=lambda p: self.evaluator.score(p.id, program_store))
        return best.id

    def select_parent_for_generation(self, program_store: ProgramStore) -> ProgramCandidate | None:
        valid_programs = program_store.get_valid_programs()
        if not valid_programs:
            return None
        return max(valid_programs, key=lambda p: self.evaluator.score(p.id, program_store))


# ============================================================================
# Environment Understanding
# ============================================================================

def _parse_environment_belief(response: str, previous_belief: EnvironmentBelief) -> EnvironmentBelief:
    """Parse the fixed KNOWN_RULES/UNCERTAIN_RULES/REWARD_RULES/STRATEGY_NOTES
    section format requested by ``ENVIRONMENT_UNDERSTANDING_PROMPT_TEMPLATE``.
    Falls back to the previous belief if the response doesn't match at all
    (e.g. the LLM replied conversationally instead of following the format)."""
    sections: dict[str, list[str]] = {
        "KNOWN_RULES": [], "UNCERTAIN_RULES": [], "REWARD_RULES": [], "STRATEGY_NOTES": [],
    }
    current: str | None = None
    for line in response.splitlines():
        stripped = line.strip()
        header = stripped.rstrip(":")
        if header in sections:
            current = header
            continue
        if current is not None and stripped.startswith("-"):
            item = stripped[1:].strip()
            if item and item.lower() != "none yet":
                sections[current].append(item)

    if not any(sections.values()):
        return previous_belief

    return EnvironmentBelief(
        known_rules=sections["KNOWN_RULES"],
        uncertain_rules=sections["UNCERTAIN_RULES"],
        reward_rules=sections["REWARD_RULES"],
        useful_strategy_notes=sections["STRATEGY_NOTES"],
    )


class EnvironmentUnderstanding:
    """Experience -> ``EnvironmentBelief``. Never generates policy code.

    Buffers transitions and only calls the LLM once ``update_every`` new
    ones have accumulated, so belief updates happen periodically rather
    than on every single step. Disabled entirely with ``enabled=False``, in
    which case ``update`` is a no-op and the belief stays empty.

    Future variants (not implemented yet):
        - none (this is what ``enabled=False`` gives you)
        - natural-language model (implemented here)
        - symbolic rule model
        - neural predictive model
        - programmatic transition model
        - ensemble world model
    """

    def __init__(
        self,
        client: LLMClient | None,
        observation_space,
        action_space,
        enabled: bool = True,
        update_every: int = 20,
    ) -> None:
        self.enabled = enabled and client is not None
        self.update_every = max(1, update_every)
        self._pending_transitions: list[Transition] = []
        self.chat: ChatSession | None = None
        if self.enabled:
            system_prompt = ENVIRONMENT_UNDERSTANDING_SYSTEM_PROMPT.format(
                observation_space_description=describe_observation_space(observation_space),
                action_space_description=describe_action_space(action_space),
            )
            self.chat = ChatSession(client, system=system_prompt, max_messages=1)

    def update(self, belief: EnvironmentBelief, recent_transitions: list[Transition]) -> EnvironmentBelief:
        if not self.enabled:
            return belief

        self._pending_transitions.extend(recent_transitions)
        if len(self._pending_transitions) < self.update_every:
            return belief

        transitions_to_use = self._pending_transitions
        self._pending_transitions = []

        prompt = ENVIRONMENT_UNDERSTANDING_PROMPT_TEMPLATE.format(
            current_belief=format_environment_belief(belief),
            recent_transitions=format_transitions(transitions_to_use),
        )
        try:
            response = self.chat.send(prompt)
        except Exception:
            return belief
        return _parse_environment_belief(response, belief)


# ============================================================================
# Hypothesis Manager
# ============================================================================

class HypothesisManager:
    """Owns falsifiable hypotheses. Mostly a placeholder for now - the agent
    functions correctly with ``enabled=False`` (``get_active_hypotheses``
    then always returns ``[]``, so hypotheses never enter prompts).

    A hypothesis is NOT a program - see ``Hypothesis``'s docstring.

    Future functionality (not implemented yet):
        - propose falsifiable hypotheses from unexplained transitions
        - attach supporting/contradicting evidence
        - update confidence
        - detect contradictions
        - choose decision-relevant hypotheses
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.hypotheses: dict[int, Hypothesis] = {}
        self._next_id = 0

    def add_hypothesis(self, claim: str, confidence: float = 0.5) -> Hypothesis:
        hypothesis_id = self._next_id
        self._next_id += 1
        hypothesis = Hypothesis(id=hypothesis_id, claim=claim, confidence=confidence)
        self.hypotheses[hypothesis_id] = hypothesis
        return hypothesis

    def get_active_hypotheses(self) -> list[Hypothesis]:
        if not self.enabled:
            return []
        return [h for h in self.hypotheses.values() if h.status != "rejected"]

    def update_hypothesis(self, hypothesis_id: int, **fields: Any) -> None:
        hypothesis = self.hypotheses.get(hypothesis_id)
        if hypothesis is None:
            return
        for key, value in fields.items():
            setattr(hypothesis, key, value)


# ============================================================================
# Experiment Planner
# ============================================================================

class ExperimentPlanner:
    """hypothesis + current knowledge -> desired experiment/behavior.

    Placeholder: the baseline agent does not depend on this, and ``plan``
    always returns ``None`` for now.

    Future variants (not implemented yet):
        - no directed experiments (what's implemented now)
        - LLM experiment planning
        - information-gain-driven planning
        - uncertainty reduction
        - falsification-focused exploration
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def plan(self, hypothesis: Hypothesis | None, environment_belief: EnvironmentBelief | None) -> ExperimentPlan | None:
        if not self.enabled or hypothesis is None:
            return None
        return None


# ============================================================================
# Knowledge Consolidator (placeholder)
# ============================================================================

class KnowledgeConsolidator:
    """Placeholder for a future cross-cutting memory step.

    For now, memory lives entirely in ``EnvironmentBelief``,
    ``HypothesisManager``, ``ExperienceStore``, and ``ProgramStore`` as
    separate systems - this class is not instantiated by the baseline agent.

    Future memory strategies (not implemented yet):
        - no persistent semantic memory (current state)
        - rolling summary
        - structured rules
        - evidence-backed rules
        - episodic + semantic memory
        - retrieval-based memory
    """


# ============================================================================
# Main Agent
# ============================================================================

# Number of recent transitions shown to the LLM when generating a candidate.
_RECENT_TRANSITIONS_FOR_GENERATION = 20


class ProgrammaticScientistAgent(BaseAgent):
    """Learns an executable programmatic policy through a small research loop
    of generation, validation, execution, and evaluation.

    Component overview (see each class's docstring for details/future variants):
        experience_store:          objective (s, a, r, s') records.
        program_store:              candidate programs + their compiled callables.
        validator:                  static + offline-behavioral candidate checks.
        generator:                  LLM-based candidate program writer.
        evaluator:                  experience -> program quality score.
        execution_manager:          how long the active program runs before re-search.
        search_controller:          which program to run/parent next (hill climbing).
        environment_understanding:  experience -> EnvironmentBelief (optional).
        hypothesis_manager:         falsifiable claims (mostly a placeholder).
        experiment_planner:         hypothesis -> directed experiment (placeholder).

    A failed candidate never destroys the current working program: a new
    source is only installed as the active program after
    ``ProgramValidator.validate`` reports ``valid=True``.

    Args:
        observation_space: The environment's observation space (for prompts).
        action_space: The environment's action space.
        client: LLM client shared by the generator and (if enabled) the
            environment-understanding component.
        steps_per_program: Fixed step budget a program gets before the
            search controller is asked to pick/generate the next one.
        max_generation_retries: Extra attempts (beyond the first) to produce
            a *valid* candidate at a given search point, each retry telling
            the LLM what was wrong with the previous attempt.
        use_environment_understanding: If ``False``, the environment belief
            stays empty and no extra LLM calls are made for it.
        use_hypotheses: If ``False``, ``HypothesisManager`` is inert and no
            hypotheses are ever attached to generation context.
        verbose: Whether to print concise debug info.
        device: Device hint (for compatibility with other agents); unused.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        client: LLMClient,
        steps_per_program: int = 20,
        max_generation_retries: int = 2,
        use_environment_understanding: bool = True,
        use_hypotheses: bool = False,
        verbose: bool = False,
        device: str = "cpu",
    ):
        super().__init__(action_space, verbose=verbose)
        self.observation_space = observation_space
        self.device = device
        self.max_generation_retries = max(0, max_generation_retries)

        # Components. Any one of these can be swapped for an alternative
        # implementation later without changing the rest of the agent, as
        # long as it exposes the same methods used below.
        self.experience_store = ExperienceStore()
        self.program_store = ProgramStore()
        self.validator = ProgramValidator(action_space)
        self.generator = ProgramGenerator(client, observation_space, action_space)
        self.evaluator = ProgramEvaluator()
        self.execution_manager = ExecutionManager(steps_per_program=steps_per_program)
        self.search_controller = SearchController(self.evaluator)
        self.environment_understanding = EnvironmentUnderstanding(
            client, observation_space, action_space, enabled=use_environment_understanding,
        )
        self.hypothesis_manager = HypothesisManager(enabled=use_hypotheses)
        self.experiment_planner = ExperimentPlanner(enabled=False)

        # A single underlying client is reused by the generator and the
        # environment-understanding component; list it once so BaseAgent's
        # usage accounting (which already dedups by id) doesn't need to.
        self.llm_clients = [client]

        # Persistent belief, carried across episodes.
        self.environment_belief = EnvironmentBelief()

        # Orchestration state.
        self.active_program_id: int | None = None
        self.current_rollout_id: int | None = None
        self.pending_observation: Any = None
        self.pending_proposed_action: Any = None
        self.pending_executed_action: Any = None
        self.pending_execution_error: str | None = None

        self.episode_id = 0
        self.episode_step = 0
        self.episode_return = 0.0
        # Steps accumulated under the current active program; may span more
        # than one episode, since ExecutionManager's budget is a total step
        # count rather than a per-episode one.
        self.steps_with_active_program = 0

        # Lifetime diagnostics for get_episode_data().
        self.num_generation_attempts = 0
        self.num_validation_failures = 0
        self.num_execution_errors = 0

        if self.verbose:
            print(f"[ProgrammaticScientist] Initialized (steps_per_program={steps_per_program})")

    # ------------------------------------------------------------------
    # Core agent interface
    # ------------------------------------------------------------------

    def select_action(self, observation: Any) -> Any:
        self._ensure_active_program(observation)

        proposed_action, error = self._execute_active_program(observation)
        if error is not None:
            executed_action = self.action_space.sample()
            self.num_execution_errors += 1
            if self.verbose:
                print(f"[ProgrammaticScientist] P{self.active_program_id} execution error: "
                      f"{error}; falling back to random action {executed_action}")
        else:
            executed_action = normalize_action(self.action_space, proposed_action)

        self.pending_observation = observation
        self.pending_proposed_action = proposed_action
        self.pending_executed_action = executed_action
        self.pending_execution_error = error

        self.episode_step += 1
        self.steps_with_active_program += 1
        return executed_action

    def update(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        if self.pending_observation is None:
            # Defensive: update() called without a matching select_action().
            return

        transition = self.experience_store.add_transition(
            rollout_id=self.current_rollout_id,
            observation=self.pending_observation,
            proposed_action=self.pending_proposed_action,
            executed_action=self.pending_executed_action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            program_id=self.active_program_id,
            episode_id=self.episode_id,
            execution_error=self.pending_execution_error,
        )
        self.episode_return += reward

        if self.active_program_id is not None:
            self.evaluator.update_from_transition(self.active_program_id, self.program_store, transition)

        if self.environment_understanding.enabled:
            self.environment_belief = self.environment_understanding.update(
                self.environment_belief, [transition],
            )

        # Hypothesis proposal/update from unexplained transitions is future
        # work; HypothesisManager.enabled currently just gates an empty hook.

        if done:
            self._finish_current_rollout()
            self.episode_id += 1

        self.pending_observation = None
        self.pending_proposed_action = None
        self.pending_executed_action = None
        self.pending_execution_error = None

    def reset(self) -> None:
        """Reset episode-specific state only. Lifetime learning (program
        store, experience store, environment belief, hypotheses, search
        statistics) persists across episodes; so does the active program
        unless a search point at start-of-episode swaps it out."""
        self.episode_step = 0
        self.episode_return = 0.0
        self.pending_observation = None
        self.pending_proposed_action = None
        self.pending_executed_action = None
        self.pending_execution_error = None
        # Defensive: normally update() already closed the rollout when done=True.
        self._finish_current_rollout()

        if self.verbose:
            print(f"[ProgrammaticScientist] Episode reset (episode {self.episode_id})")

    def get_episode_data(self) -> dict[str, Any]:
        all_programs = self.program_store.get_all_programs()
        valid_programs = self.program_store.get_valid_programs()

        artifacts: dict[str, Any] = {
            "environment_belief.txt": format_environment_belief(self.environment_belief),
            "hypotheses.txt": format_hypotheses(list(self.hypothesis_manager.hypotheses.values())),
            "program_search.txt": format_program_search(all_programs),
        }
        for program in all_programs:
            artifacts[f"programs/program_{program.id}.py"] = program.source

        return {
            "metrics": {
                "episode_steps": self.episode_step,
                "episode_return": self.episode_return,
                "active_program_id": self.active_program_id,
                "num_programs_total": len(all_programs),
                "num_valid_programs": len(valid_programs),
                "num_generation_attempts": self.num_generation_attempts,
                "num_validation_failures": self.num_validation_failures,
                "num_execution_errors": self.num_execution_errors,
            },
            "artifacts": artifacts,
        }

    # ------------------------------------------------------------------
    # select_action() helpers
    # ------------------------------------------------------------------

    def _ensure_active_program(self, observation: Any) -> None:
        """Make sure there is an active program (searching/generating one if
        needed), without doing any environment-model updates here - those
        only happen once a real transition is observed, in update()."""
        if self.active_program_id is not None and self.current_rollout_id is None:
            # New episode, same active program: just open a fresh rollout.
            self.current_rollout_id = self.experience_store.start_rollout(self.active_program_id)
            self.program_store.record_rollout(self.active_program_id, self.current_rollout_id)
            return

        if self.active_program_id is None:
            self._maybe_search_or_generate(observation, reason="no active program yet")
            return

        if self.execution_manager.should_stop(
            self.active_program_id, self.steps_with_active_program, self.current_rollout_id
        ):
            self._maybe_search_or_generate(observation, reason="execution budget exhausted")

    def _maybe_search_or_generate(self, observation: Any, reason: str) -> None:
        """One search point: close out the current rollout, try to generate
        one improved candidate, then ask the search controller which
        (existing, now possibly-including-the-new-candidate) program to run
        next."""
        self._finish_current_rollout()

        parent_program = self.search_controller.select_parent_for_generation(self.program_store)
        self._generate_and_register_candidate(parent_program, reason)

        next_program_id = self.search_controller.select_program_to_execute(self.program_store)

        if (next_program_id is not None and next_program_id != self.active_program_id
                and self.verbose):
            print(f"[ProgrammaticScientist] Switching active program "
                  f"{self.active_program_id} -> {next_program_id}")

        self.active_program_id = next_program_id
        self.steps_with_active_program = 0
        if next_program_id is not None:
            self.current_rollout_id = self.experience_store.start_rollout(next_program_id)
            self.program_store.record_rollout(next_program_id, self.current_rollout_id)
        else:
            # No valid program exists at all (yet); random-action fallback
            # kicks in naturally via _execute_active_program below.
            self.current_rollout_id = None

    def _generate_and_register_candidate(
        self, parent_program: ProgramCandidate | None, reason: str
    ) -> ProgramCandidate | None:
        """Generate -> validate -> (only if valid) install as a candidate.
        Retries up to ``max_generation_retries`` extra times, feeding back
        the previous error. Never touches ``active_program_id`` - the caller
        decides whether/when to switch to a newly valid candidate."""
        context = ProgramGenerationContext(
            parent_program=parent_program,
            recent_transitions=self.experience_store.get_recent_transitions(
                _RECENT_TRANSITIONS_FOR_GENERATION
            ),
            environment_belief=self.environment_belief if self.environment_understanding.enabled else None,
            hypotheses=self.hypothesis_manager.get_active_hypotheses(),
            generation_reason=reason,
        )
        parent_id = parent_program.id if parent_program is not None else None
        hypothesis_ids = [h.id for h in context.hypotheses]

        last_error: str | None = None
        for attempt in range(self.max_generation_retries + 1):
            context.retry_feedback = last_error
            self.num_generation_attempts += 1
            try:
                source = self.generator.generate(context)
            except Exception as e:
                last_error = f"LLM generation failed: {e}"
                if self.verbose:
                    print(f"[ProgrammaticScientist] Generation attempt {attempt + 1} failed: {last_error}")
                continue

            validation_result = self.validator.validate(source, self.experience_store)
            candidate = self.program_store.add_program(
                source=source,
                parent_id=parent_id,
                generation_reason=reason,
                hypothesis_ids=hypothesis_ids,
            )

            if validation_result.valid:
                self.program_store.mark_valid(candidate.id, validation_result.policy_fn)
                if self.verbose:
                    parent_desc = f"P{parent_id}" if parent_id is not None else "scratch"
                    print(f"[ProgrammaticScientist] Generated candidate P{candidate.id} from "
                          f"{parent_desc}; validation passed")
                return candidate

            self.num_validation_failures += 1
            self.program_store.mark_invalid(candidate.id, validation_result.error)
            last_error = validation_result.error
            if self.verbose:
                print(f"[ProgrammaticScientist] Candidate P{candidate.id} validation failed: "
                      f"{validation_result.error}")

        if self.verbose:
            print(f"[ProgrammaticScientist] Exhausted {self.max_generation_retries + 1} "
                  f"generation attempt(s) for this search point; keeping the current program.")
        return None

    def _execute_active_program(self, observation: Any) -> tuple[Any, str | None]:
        """Run the active program's policy on ``observation``.

        Returns ``(proposed_action, error)``. ``proposed_action`` is
        whatever the policy returned (or ``None`` if it raised), kept
        separate from any later random fallback so a fallback's outcome is
        never credited to the program.
        """
        if self.active_program_id is None:
            return None, "No active program."

        policy_fn = self.program_store.get_policy(self.active_program_id)
        if policy_fn is None:
            return None, "Active program has no compiled policy."

        try:
            proposed_action = policy_fn(observation)
        except Exception as e:
            return None, f"Policy execution raised: {e}"

        if not is_valid_action(self.action_space, proposed_action):
            return proposed_action, f"Policy returned invalid action: {proposed_action!r}"

        return proposed_action, None

    def _finish_current_rollout(self) -> None:
        if self.current_rollout_id is None:
            return
        rollout = self.experience_store.finish_rollout(self.current_rollout_id)
        if rollout is not None and rollout.program_id is not None:
            self.evaluator.update_from_rollout(rollout.program_id, self.program_store, rollout)
            if self.verbose:
                print(f"[ProgrammaticScientist] Finished rollout {rollout.id}: "
                      f"return={rollout.total_return:.2f}")
        self.current_rollout_id = None
