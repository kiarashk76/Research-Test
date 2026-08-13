"""ProgrammaticLLMAgent: LLM periodically writes a Python policy program.

Instead of asking the LLM for individual actions, this agent asks it to write
a small ``policy(observation) -> action`` function every ``n_actions`` steps.
The generated program is executed locally (in a restricted globals dict) to
pick actions until the next regeneration point.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Optional

import numpy as np

from llm import LLMClient, ChatSession

from .base import BaseAgent


# ============================================================================
# PROMPTS (at top as constants with replaceable parts)
# ============================================================================

SYSTEM_PROMPT = """You are a programmer writing control policies for an agent in a \
reinforcement learning environment.

Observation space: {observation_space_description}
Available actions: {action_space_description}

Instead of choosing actions directly, you write a Python program that defines a \
policy function. That function is called once per environment step, receives the \
raw observation described above, and must return a valid action."""

PROGRAM_GENERATION_PROMPT_TEMPLATE = """{learned_knowledge}Write (or improve) the policy program for this environment.

Observation space: {observation_space_description}
Action space: {action_space_description}

Current program (or "None yet" if this is the first one):
{current_program}

Trajectory since the last program was generated:
{trajectory_text}

Current observation: {current_observation}

Requirements:
- Return ONLY raw Python source code. No Markdown, no code fences, no commentary.
- Define exactly one required entry point:
    def policy(observation):
        ...
        return action
- The returned action must be valid for the action space described above.
- You may use ordinary Python logic: loops, conditionals, helper functions, local constants.
- Use the trajectory, program errors, learned knowledge, and rewards above to improve on the \
current program (or write a new one if none exists or the previous one failed).
- Do not access files, network, environment variables, subprocesses, or any external resource.
- Imports are not available (no "import" statements at all, not even for numpy or math). \
`np` (numpy) and `math` are already provided as globals, ready to use directly, e.g. `np.array(...)`.
- Prefer deterministic, concise programs unless randomness is strategically useful.

Respond with only the Python source code."""

KNOWLEDGE_UPDATE_PROMPT_TEMPLATE = """You are learning about an environment through episodes of \
programmatic policies. Update your knowledge base based on the new episode.

Previous knowledge (or empty if this is the first episode):
{previous_knowledge}

New episode trajectory:
Observations: {observations}
Actions taken: {actions}
Rewards received: {rewards}
Total return: {total_return}

Programs generated during this episode (in order, with any execution errors observed while they \
were active):
{programs}

Update the knowledge base with the following structure:
- **Environment Rules**: What are the deterministic rules governing state transitions?
- **Successful Strategies**: What action patterns led to high rewards?
- **Failed Strategies**: What action patterns led to low rewards or should be avoided?
- **Programmatic Policy Lessons**: What made a generated program work or fail (bugs, bad \
assumptions, wrong action format, etc.)?
- **Remaining Hypotheses**: What questions or uncertainties remain?
- **Recommended Strategy**: What is the best strategy for the next episode based on what you've learned?

Preserve useful previous knowledge. Revise or correct beliefs that are contradicted by this new episode.
Be concise but precise."""


# ============================================================================
# Restricted builtins available inside a generated program
# ============================================================================

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "map": map, "filter": filter, "isinstance": isinstance,
    "True": True, "False": False, "None": None,
}


class ProgrammaticLLMAgent(BaseAgent):
    """LLM-based agent whose policy is a generated Python program.

    Every ``n_actions`` steps, the LLM is asked to (re)write a
    ``policy(observation) -> action`` program based on persistent learned
    knowledge, the current program, and the trajectory collected since the
    last generation. The program is executed locally at every step; on any
    failure (exception, invalid action) a random action is taken instead and
    the failure is recorded as feedback for the next generation.

    Args:
        observation_space: The environment's observation space (for context).
        action_space: The environment's action space.
        n_actions: Number of steps a generated program is used for before
            regeneration (default 10).
        client: Optional LLMClient instance. If None, a new one is created
                with default settings.
        verbose: Whether to print debug info.
        device: Device hint (for compatibility with other agents).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        client: LLMClient,
        n_actions: int = 10,
        verbose: bool = False,
        device: str = "cpu",
    ):
        super().__init__(action_space, verbose=verbose)
        self.observation_space = observation_space
        self.n_actions = max(1, n_actions)
        self.device = device

        self.client = client
        system_prompt = SYSTEM_PROMPT.format(
            observation_space_description=self._describe_observation_space(),
            action_space_description=self._describe_action_space(),
        )
        self.chat = ChatSession(
            self.client,
            system=system_prompt,
            max_messages=1,  # Keep only recent messages for independent calls
        )

        # Current program state
        self.current_program: Optional[str] = None
        self.policy_function: Optional[Callable[[Any], Any]] = None
        self.compile_error: Optional[str] = None

        # Persistent knowledge learned across episodes
        self.learned_knowledge = ""

        # Full-episode history (kept until reset())
        self.observation_history: list[Any] = []
        self.action_history: list[Any] = []
        self.reward_history: list[float] = []
        self.program_error_history: list[Optional[str]] = []
        self.program_versions: list[str] = []

        # Trajectory since the last program generation (cleared on regeneration)
        self.recent_observations: list[Any] = []
        self.recent_actions: list[Any] = []
        self.recent_rewards: list[float] = []
        self.recent_program_errors: list[Optional[str]] = []

        self.step_count = 0
        self.program_generation_count = 0
        self.steps_since_program_generation = 0

        if self.verbose:
            print(f"[ProgrammaticLLMAgent] Initialized with n_actions={self.n_actions}")

    # ------------------------------------------------------------------
    # Core agent interface
    # ------------------------------------------------------------------

    def select_action(self, observation: Any) -> Any:
        """Select an action, regenerating the policy program if needed."""
        if self.current_program is None or self.steps_since_program_generation >= self.n_actions:
            self._generate_new_program(observation)
            self.steps_since_program_generation = 0
            
        self.observation_history.append(observation)
        self.recent_observations.append(observation)

        action, error = self._execute_policy(observation)
        if error is not None:
            action = self.action_space.sample()
            if self.verbose:
                print(f"[ProgrammaticLLMAgent] Policy execution failed ({error}); "
                      f"using fallback random action {action}")

        self.action_history.append(action)
        self.program_error_history.append(error)
        self.recent_actions.append(action)
        self.recent_program_errors.append(error)

        self.step_count += 1
        self.steps_since_program_generation += 1
        return action

    def update(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        """Receive feedback after an action and update knowledge when done."""
        self.reward_history.append(reward)
        self.recent_rewards.append(reward)

        if done:
            self._update_learned_knowledge()

    def reset(self) -> None:
        """Reset episode-specific state; keep persistent learned knowledge."""
        self.observation_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.program_error_history.clear()
        self.program_versions.clear()

        self.step_count = 0

        if self.verbose:
            print("[ProgrammaticLLMAgent] Episode reset")

    def get_episode_data(self) -> dict[str, Any]:
        """Return episode data for logging."""
        num_errors = sum(1 for e in self.program_error_history if e is not None)
        return {
            "metrics": {
                "total_steps": self.step_count,
                "total_program_generations": self.program_generation_count,
                "program_execution_errors": num_errors,
                "episode_return": sum(self.reward_history),
            },
            "artifacts": {
                "learned_knowledge.txt": self.learned_knowledge if self.learned_knowledge else "No knowledge learned yet.",
                "final_program.py": self.current_program if self.current_program else "# No program generated yet.",
                "program_errors.txt": "\n".join(
                    f"Step {i}: {e}" for i, e in enumerate(self.program_error_history) if e is not None
                ) or "No program execution errors.",
            },
        }

    # ------------------------------------------------------------------
    # Program generation
    # ------------------------------------------------------------------

    def _generate_new_program(self, current_observation: Any) -> None:
        """Query the LLM for a new policy program and compile it."""
        if self.verbose:
            print(f"[ProgrammaticLLMAgent] Generating new program "
                  f"(step {self.step_count}, generation {self.program_generation_count})")

        learned_knowledge_context = ""
        if self.learned_knowledge:
            learned_knowledge_context = f"Your current knowledge:\n{self.learned_knowledge}\n\n"

        prompt = PROGRAM_GENERATION_PROMPT_TEMPLATE.format(
            learned_knowledge=learned_knowledge_context,
            observation_space_description=self._describe_observation_space(),
            action_space_description=self._describe_action_space(),
            current_program=self.current_program if self.current_program else "None yet.",
            trajectory_text=self._format_recent_trajectory(),
            current_observation=self._format_observation(current_observation),
        )

        try:
            response = self.chat.send(prompt)
            source = self._clean_program_source(response)
            self._compile_program(source)
        except Exception as e:
            self.current_program = None
            self.policy_function = None
            self.compile_error = f"LLM query failed: {e}"
            if self.verbose:
                print(f"[ProgrammaticLLMAgent] Error generating program: {e}")

        self.program_generation_count += 1
        if self.current_program:
            self.program_versions.append(self.current_program)

        # Start a fresh trajectory window for this program
        self.recent_observations.clear()
        self.recent_actions.clear()
        self.recent_rewards.clear()
        self.recent_program_errors.clear()

    def _clean_program_source(self, response: str) -> str:
        """Strip accidental Markdown code fences from the LLM response."""
        text = response.strip()
        fence_match = re.match(r"^```[ \t]*\w*[ \t]*\r?\n(.*)\r?\n```$", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        return text

    def _compile_program(self, program_source: str) -> None:
        """Compile the generated source and extract the ``policy`` callable.

        On success, sets ``self.current_program`` and ``self.policy_function``.
        On failure, keeps the source for feedback but sets the executable
        policy to ``None`` so callers fall back to random actions.
        """
        self.current_program = program_source
        self.compile_error = None

        restricted_globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "np": np,
            "math": math,
        }
        try:
            exec(compile(program_source, "<generated_policy>", "exec"), restricted_globals)
        except Exception as e:
            self.policy_function = None
            self.compile_error = f"Compilation failed: {e}"
            if self.verbose:
                print(f"[ProgrammaticLLMAgent] {self.compile_error}")
            return

        policy_fn = restricted_globals.get("policy")
        if not callable(policy_fn):
            self.policy_function = None
            self.compile_error = "Generated program does not define a callable 'policy(observation)' function."
            if self.verbose:
                print(f"[ProgrammaticLLMAgent] {self.compile_error}")
            return

        self.policy_function = policy_fn

    def _execute_policy(self, observation: Any) -> tuple[Any, Optional[str]]:
        """Run the current policy on ``observation``.

        Returns:
            (action, error): ``error`` is ``None`` on success.
        """
        if self.policy_function is None:
            return None, self.compile_error or "No policy function available."

        try:
            action = self.policy_function(observation)
        except Exception as e:
            return None, f"Policy execution raised: {e}"

        if not self._is_valid_action(action):
            return None, f"Policy returned invalid action: {action!r}"

        return action, None

    def _is_valid_action(self, action: Any) -> bool:
        """Validate an action against the action space, with light normalization."""
        try:
            if self.action_space.contains(action):
                return True
        except Exception:
            pass

        # Discrete action spaces: allow integer-like values (e.g. numpy ints, bools).
        if hasattr(self.action_space, "n"):
            try:
                normalized = int(action)
            except (TypeError, ValueError):
                return False
            try:
                return self.action_space.contains(normalized)
            except Exception:
                return 0 <= normalized < self.action_space.n

        return False

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_observation(self, obs: Any) -> str:
        """Format an observation for inclusion in the prompt."""
        if isinstance(obs, (list, tuple)):
            return str(list(obs)[:10])  # Truncate long lists
        elif isinstance(obs, dict):
            return str({k: v for i, (k, v) in enumerate(obs.items()) if i < 5})
        else:
            return str(obs)

    def _describe_observation_space(self) -> str:
        """Describe the observation space for the LLM."""
        space = self.observation_space
        if space is None:
            return "Unknown (not provided); infer structure from the observations shown below."
        try:
            # Handle gym.spaces.Box (continuous / array-like)
            if hasattr(space, "shape") and hasattr(space, "low"):
                return f"Array of shape {space.shape} with values in range [{space.low}, {space.high}]"
            # Handle gym.spaces.Discrete
            elif hasattr(space, "n"):
                return f"Discrete: integers from 0 to {space.n - 1}"
            # Handle gym.spaces.Dict
            elif hasattr(space, "spaces") and isinstance(getattr(space, "spaces"), dict):
                fields = ", ".join(space.spaces.keys())
                return f"Dict with fields: {fields}"
            else:
                return str(space)
        except Exception:
            return str(space)

    def _describe_action_space(self) -> str:
        """Describe the action space for the LLM."""
        try:
            # Handle gym.spaces.Discrete
            if hasattr(self.action_space, "n"):
                return f"Discrete actions: integers from 0 to {self.action_space.n - 1}"
            # Handle gym.spaces.Box (continuous)
            elif hasattr(self.action_space, "shape") and hasattr(self.action_space, "low"):
                low = self.action_space.low
                high = self.action_space.high
                shape = self.action_space.shape
                return f"Continuous actions: array of shape {shape} with values in range [{low}, {high}]"
            # Fallback
            else:
                return str(self.action_space)
        except Exception:
            return str(self.action_space)

    def _format_recent_trajectory(self) -> str:
        """Format the trajectory collected since the last program generation."""
        if not self.recent_observations:
            return "No trajectory yet (this is the first program)."

        lines = []
        for i, obs in enumerate(self.recent_observations):
            obs_str = self._format_observation(obs)
            line = f"  Step {i}: Observation = {obs_str}"
            if i < len(self.recent_actions):
                line += f", Action = {self.recent_actions[i]}"
            if i < len(self.recent_rewards):
                line += f", Reward = {self.recent_rewards[i]:.2f}"
            if i < len(self.recent_program_errors) and self.recent_program_errors[i]:
                line += f", Error = {self.recent_program_errors[i]}"
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Knowledge update
    # ------------------------------------------------------------------

    def _update_learned_knowledge(self) -> None:
        """Update learned knowledge based on the full episode trajectory."""
        if not self.observation_history or not self.action_history:
            return

        if self.verbose:
            print(f"[ProgrammaticLLMAgent] Updating knowledge for episode "
                  f"(steps={len(self.action_history)}, reward={sum(self.reward_history):.2f})")

        observations_str = "\n".join(
            f"  Step {i}: {self._format_observation(obs)}"
            for i, obs in enumerate(self.observation_history)
        )
        actions_str = "\n".join(
            f"  Step {i}: {action}" for i, action in enumerate(self.action_history)
        )
        rewards_str = "\n".join(
            f"  Step {i}: {reward:.2f}" for i, reward in enumerate(self.reward_history)
        )
        total_return = sum(self.reward_history)

        errors_str = "\n".join(
            f"  Step {i}: {e}" for i, e in enumerate(self.program_error_history) if e
        ) or "  No execution errors."
        programs_str = "\n\n".join(
            f"--- Program v{i + 1} ---\n{program}"
            for i, program in enumerate(self.program_versions)
        ) or "  No programs were successfully compiled."
        programs_str += f"\n\nExecution errors observed:\n{errors_str}"

        prompt = KNOWLEDGE_UPDATE_PROMPT_TEMPLATE.format(
            previous_knowledge=self.learned_knowledge if self.learned_knowledge else "No prior knowledge yet.",
            observations=observations_str,
            actions=actions_str,
            rewards=rewards_str,
            total_return=f"{total_return:.2f}",
            programs=programs_str,
        )

        try:
            updated_knowledge = self.chat.send(prompt)
            self.learned_knowledge = updated_knowledge

            if self.verbose:
                print(f"[ProgrammaticLLMAgent] Knowledge updated: {updated_knowledge[:150]}...")
        except Exception as e:
            if self.verbose:
                print(f"[ProgrammaticLLMAgent] Error updating knowledge: {e}")
