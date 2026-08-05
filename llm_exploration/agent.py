from __future__ import annotations

import random
import re
import traceback
from collections import defaultdict
from typing import Any

import numpy as np

from llm import ChatSession, LLMClient, extract_code
from llm.test_api import API_KEY, BASE_URL, MODEL


class SimpleAgent:
    """Minimal agent with the standard act/update shape."""

    def __init__(self, action_space) -> None:
        self.action_space = action_space
        self.history: list[dict[str, Any]] = []

    def act(self, observation) -> int:
        """Choose an action for the current observation."""
        return int(self.action_space.sample())

    def update(self, observation, action: int, reward: float, next_observation, done: bool) -> None:
        """Receive one transition from the environment."""
        self.history.append(
            {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "done": done,
            }
        )


class QLearningAgent:
    """Small tabular Q-learning agent for discrete actions."""

    def __init__(
        self,
        action_space,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
    ) -> None:
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_values = defaultdict(lambda: np.zeros(self.action_space.n, dtype=float))

    def act(self, observation) -> int:
        state = self._state_key(observation)
        if random.random() < self.epsilon:
            return int(self.action_space.sample())
        return int(np.argmax(self.q_values[state]))

    def update(self, observation, action: int, reward: float, next_observation, done: bool) -> None:
        state = self._state_key(observation)
        next_state = self._state_key(next_observation)

        target = float(reward)
        if not done:
            target += self.discount * float(np.max(self.q_values[next_state]))

        old_value = self.q_values[state][action]
        self.q_values[state][action] = old_value + self.learning_rate * (target - old_value)

    def _state_key(self, observation) -> tuple:
        return tuple(np.asarray(observation, dtype=int).reshape(-1).tolist())


class LLMAgent:
    """LLM-driven agent with text understanding and plan memory."""

    def __init__(
        self,
        action_space,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_steps_for_plan: int = 20,
        system_prompt: str | None = None,
    ) -> None:
        self.action_space = action_space
        self.Understand = ""
        self.Plan = ""
        self.max_steps_for_plan = max_steps_for_plan
        self.steps_in_plan = 0
        self.history: list[str] = []
        client = LLMClient(
            model=model or MODEL,
            api_key=api_key or API_KEY,
            base_url=base_url or BASE_URL,
            timeout=30,
            max_retries=2,
            stream=True,
            temperature=0.1,
        )
        self.chat = ChatSession(
            client,
            system=system_prompt or (
                "You are an agent solving a hidden-rule reinforcement learning game. "
                "Infer the action dynamics only from observations, actions, rewards, "
                "and done flags. Keep UNDERSTANDING about dynamics only, with no goals "
                "or strategy. Keep PLAN as a concrete short strategy for getting reward. "
                "When asked for an action, return only the action integer and no other text. "
                "When asked YES or NO, return only YES or NO."
            ),
        )

    def act(self, observation) -> int:
        if not self.Plan:
            self._make_initial_plan(observation)
        
        print(f"UNDERSTANDING:\n{self.Understand}\n\nPLAN:\n{self.Plan}\n")
        
        reply = self.chat.send(
            "Given the current plan and observation, choose the next action. "
            f"Return only one integer from 0 to {self.action_space.n - 1}.\n\n"
            f"UNDERSTANDING:\n{self.Understand}\n\n"
            f"PLAN:\n{self.Plan}\n\n"
            f"OBSERVATION:\n{self._obs_text(observation)}",
            max_tokens=100,
        )
        action = self._parse_action(reply)
        if action is None:
            return int(self.action_space.sample())
        return action

    def update(self, observation, action: int, reward: float, next_observation, done: bool) -> None:
        transition = (
            f"Observation: {self._obs_text(observation)}\n"
            f"Action: {action}\n"
            f"Reward: {reward}\n"
            f"Next observation: {self._obs_text(next_observation)}\n"
            f"Done: {done}"
        )
        self.history.append(transition)
        self.steps_in_plan += 1

        plan_finished = self._plan_finished(transition)
        plan_timed_out = self.steps_in_plan >= self.max_steps_for_plan
        if plan_finished or plan_timed_out:
            reason = "finished" if plan_finished else "reached max_steps_for_plan"
            self._refresh_understanding_and_plan(reason)

    def _make_initial_plan(self, observation) -> None:
        reply = self.chat.send(
            "This is the first observation. Create an initial understanding and plan. "
            "UNDERSTANDING must only describe possible environment dynamics. "
            "PLAN is the strategy for finding reward. Use exactly these headings:\n"
            "UNDERSTANDING:\n"
            "PLAN:\n\n"
            f"OBSERVATION:\n{self._obs_text(observation)}",
            max_tokens=1500,
        )
        self.Understand, self.Plan = self._parse_memory(reply)
        self.steps_in_plan = 0

    def _plan_finished(self, transition: str) -> bool:
        reply = self.chat.send(
            "Based on the current plan and the latest transition, has the plan finished? "
            "Answer only YES or NO.\n\n"
            f"PLAN:\n{self.Plan}\n\n"
            f"Latest transition:\n{transition}",
            max_tokens=10,
        )
        return reply.strip().lower().startswith("yes")

    def _refresh_understanding_and_plan(self, reason: str) -> None:
        self.Understand = self.chat.send(
            "Update your understanding of the environment dynamics only. "
            "Do not include goals, strategy, or plans. Return only the updated "
            "understanding.\n\n"
            f"Reason for update: the current plan {reason}.\n\n"
            f"Current understanding:\n{self.Understand}\n\n"
            f"Recent history:\n{self._recent_history()}",
            max_tokens=1500,
        ).strip()
        self.Plan = self.chat.send(
            "Create a new plan for getting reward using the current understanding "
            "and recent history. Return only the plan.\n\n"
            f"Reason for new plan: the previous plan {reason}.\n\n"
            f"UNDERSTANDING:\n{self.Understand}\n\n"
            f"Recent history:\n{self._recent_history()}",
            max_tokens=1500,
        ).strip()
        self.steps_in_plan = 0

    def _parse_action(self, text: str) -> int | None:
        match = re.search(rf"\b[0-{self.action_space.n - 1}]\b", text)
        return None if match is None else int(match.group(0))

    def _parse_memory(self, text: str) -> tuple[str, str]:
        understanding_match = re.search(
            r"UNDERSTANDING:\s*(.*?)(?:\n\s*PLAN:|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        plan_match = re.search(r"PLAN:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        understanding = understanding_match.group(1).strip() if understanding_match else ""
        plan = plan_match.group(1).strip() if plan_match else text.strip()
        return understanding, plan

    def _recent_history(self, n: int = 10) -> str:
        return "\n---\n".join(self.history[-n:])

    def _obs_text(self, observation) -> str:
        return str(np.asarray(observation, dtype=int).reshape(-1).tolist())


class HybridAgent:
    """Q-learning over primitive actions plus LLM-generated programmatic options."""

    def __init__(
        self,
        action_space,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
        optimistic_initial_value: float = 5.0,
        option_request_frequency: int = 50,
        max_options: int = 10,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.optimistic_initial_value = optimistic_initial_value
        self.option_request_frequency = option_request_frequency
        self.max_options = max_options

        self.primitive_action_count = self.action_space.n
        self.options: list[dict[str, Any]] = []
        self.q_values: dict[tuple, dict[int, float]] = defaultdict(dict)
        self.history: list[str] = []
        self.total_steps = 0
        self.last_q_action: int | None = None

        client = LLMClient(
            model=model or MODEL,
            api_key=api_key or API_KEY,
            base_url=base_url or BASE_URL,
            timeout=30,
            max_retries=2,
            stream=True,
            temperature=0.1,
        )
        self.chat = ChatSession(
            client,
            system=system_prompt or (
                "You help a reinforcement learning agent solve HiddenChainEnv. "
                "The visible observation is a length-3 binary vector [A, B, C]. "
                "The agent can choose primitive actions 0, 1, 2, or 3. "
                "The objective is to discover action dynamics from history and get "
                "reward; reward is observed only after environment steps. "
                "When useful, propose programmatic options. An option must be a small "
                "deterministic Python function named option(observation, history) that "
                "returns one primitive action integer. Use only the visible observation, "
                "action/reward history, and simple Python logic. Do not assume hidden "
                "state directly."
            ),
        )

    def act(self, observation) -> int:
        state = self._state_key(observation)
        q_action = self._choose_q_action(state)
        self.last_q_action = q_action

        if q_action < self.primitive_action_count:
            return q_action
        return self._run_option(q_action, observation)

    def update(self, observation, action: int, reward: float, next_observation, done: bool) -> None:
        q_action = self.last_q_action if self.last_q_action is not None else action
        state = self._state_key(observation)
        next_state = self._state_key(next_observation)

        target = float(reward)
        if not done:
            target += self.discount * max(self._q(next_state, a) for a in self._available_actions())

        old_value = self._q(state, q_action)
        self.q_values[state][q_action] = old_value + self.learning_rate * (target - old_value)

        self.total_steps += 1
        self.history.append(
            f"Step: {self.total_steps}\n"
            f"Observation: {self._obs_text(observation)}\n"
            f"Q action: {q_action} ({self._action_name(q_action)})\n"
            f"Env action: {action}\n"
            f"Reward: {reward}\n"
            f"Next observation: {self._obs_text(next_observation)}\n"
            f"Done: {done}"
        )

        if self.total_steps % self.option_request_frequency == 0:
            self._maybe_add_option()

    def _choose_q_action(self, state: tuple) -> int:
        actions = self._available_actions()
        if random.random() < self.epsilon:
            return random.choice(actions)
        return max(actions, key=lambda action: self._q(state, action))

    def _available_actions(self) -> list[int]:
        return list(range(self.primitive_action_count + len(self.options)))

    def _q(self, state: tuple, action: int) -> float:
        if action not in self.q_values[state]:
            self.q_values[state][action] = self._initial_q(action)
        return self.q_values[state][action]

    def _initial_q(self, action: int) -> float:
        if action < self.primitive_action_count:
            return 0.0
        return self.optimistic_initial_value

    def _run_option(self, q_action: int, observation) -> int:
        option_idx = q_action - self.primitive_action_count
        option = self.options[option_idx]

        try:
            return self._call_option(option, observation)
        except Exception:
            error = traceback.format_exc()
            self._repair_option(option_idx, error, observation)

        try:
            return self._call_option(self.options[option_idx], observation)
        except Exception:
            return int(self.action_space.sample())

    def _call_option(self, option: dict[str, Any], observation) -> int:
        action = option["fn"](np.asarray(observation, dtype=int).copy(), list(self.history))
        action = int(action)
        if action < 0 or action >= self.primitive_action_count:
            raise ValueError(
                f"Option returned invalid action {action}; expected 0..{self.primitive_action_count - 1}"
            )
        return action

    def _maybe_add_option(self) -> None:
        if len(self.options) >= self.max_options:
            return

        reply = self.chat.send(
            "Given the recent history, is there an interesting repeated behavior, "
            "subgoal, or policy worth adding as a programmatic option? "
            "If no, return exactly NO. If yes, return only a Python code block defining:\n"
            "def option(observation, history):\n"
            "    ...\n"
            f"The function must return one primitive action integer from 0 to {self.primitive_action_count - 1}.\n\n"
            f"Recent history:\n{self._recent_history()}",
            max_tokens=1500,
        )
        if reply.strip().upper().startswith("NO"):
            return

        code = self._extract_code(reply)
        fn = self._compile_option(code)
        if fn is None:
            repaired = self._repair_new_option(code)
            if repaired is None:
                return
            code, fn = repaired

        self.options.append(
            {
                "name": f"option_{len(self.options)}",
                "code": code,
                "fn": fn,
            }
        )

    def _repair_option(self, option_idx: int, error: str, observation) -> None:
        option = self.options[option_idx]
        reply = self.chat.send(
            "This option code failed while running. Rewrite it to fix the error. "
            "Return only a Python code block defining:\n"
            "def option(observation, history):\n"
            "    ...\n"
            f"The function must return one primitive action integer from 0 to {self.primitive_action_count - 1}.\n\n"
            f"Current observation:\n{self._obs_text(observation)}\n\n"
            f"Error:\n{error}\n\n"
            f"Broken code:\n```python\n{option['code']}\n```",
            max_tokens=1500,
        )
        code = self._extract_code(reply)
        fn = self._compile_option(code)
        if fn is None:
            return
        option["code"] = code
        option["fn"] = fn

    def _repair_new_option(self, code: str):
        _, error = self._compile_option_with_error(code)
        reply = self.chat.send(
            "This new option code failed before it could be added. Rewrite it to fix "
            "the error. Return only a Python code block defining:\n"
            "def option(observation, history):\n"
            "    ...\n"
            f"The function must return one primitive action integer from 0 to {self.primitive_action_count - 1}.\n\n"
            f"Error:\n{error}\n\n"
            f"Broken code:\n```python\n{code}\n```",
            max_tokens=1500,
        )
        repaired_code = self._extract_code(reply)
        fn = self._compile_option(repaired_code)
        if fn is None:
            return None
        return repaired_code, fn

    def _compile_option(self, code: str):
        fn, _ = self._compile_option_with_error(code)
        return fn

    def _compile_option_with_error(self, code: str):
        namespace: dict[str, Any] = {}
        safe_builtins = {
            "abs": abs,
            "bool": bool,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "sum": sum,
            "tuple": tuple,
        }
        try:
            exec(code, {"__builtins__": safe_builtins, "np": np}, namespace)
        except Exception:
            return None, traceback.format_exc()
        fn = namespace.get("option")
        if not callable(fn):
            return None, "Code did not define a callable function named option."
        return fn, ""

    def _extract_code(self, text: str) -> str:
        try:
            return extract_code(text)
        except ValueError:
            return text.strip()

    def _action_name(self, action: int) -> str:
        if action < self.primitive_action_count:
            return f"primitive_{action}"
        return self.options[action - self.primitive_action_count]["name"]

    def _recent_history(self, n: int = 50) -> str:
        return "\n---\n".join(self.history[-n:])

    def _state_key(self, observation) -> tuple:
        return tuple(np.asarray(observation, dtype=int).reshape(-1).tolist())

    def _obs_text(self, observation) -> str:
        return str(np.asarray(observation, dtype=int).reshape(-1).tolist())
