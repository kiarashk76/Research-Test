"""SimpleLLMAgent: Calls the LLM every N steps to request N actions ahead."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from llm import LLMClient, ChatSession

from .base import BaseAgent


# ============================================================================
# PROMPTS (at top as constants with replaceable parts)
# ============================================================================

SYSTEM_PROMPT = """You are a strategic agent in a reinforcement learning environment. 
Your goal is to maximize cumulative reward by selecting optimal actions based on observations.

Available actions: {action_space_description}

When asked for actions, respond with a JSON array of actions in the exact format specified.
Be concise and efficient in your reasoning."""

ACTION_REQUEST_PROMPT_TEMPLATE = """{learned_knowledge}Based on the following observation and action history, 
suggest the next {n_actions} actions to maximize reward.

{history_text}

Current observation: {current_observation}

Respond with a JSON array of exactly {n_actions} action(s) like this:
{{"actions": [action1, action2, ...]}}

Only respond with the JSON array, no additional text."""

KNOWLEDGE_UPDATE_PROMPT_TEMPLATE = """You are learning about an environment through episodes. Update your knowledge base based on the new trajectory.

Previous knowledge (or empty if this is the first episode):
{previous_knowledge}

New episode trajectory:
Observations: {observations}
Actions taken: {actions}
Rewards received: {rewards}
Total return: {total_return}

Update the knowledge base with the following structure:
- **Environment Rules**: What are the deterministic rules governing state transitions?
- **Successful Strategies**: What action patterns led to high rewards?
- **Failed Strategies**: What action patterns led to low rewards or should be avoided?
- **Remaining Hypotheses**: What questions or uncertainties remain?
- **Recommended Strategy**: What is the best strategy for the next episode based on what you've learned?

Preserve useful previous knowledge. Revise or correct beliefs that are contradicted by this new episode.
Be concise but precise."""


# ============================================================================
# SimpleLLMAgent
# ============================================================================

class SimpleLLMAgent(BaseAgent):
    """LLM-based agent that requests N actions ahead every N steps.
    
    Every N steps, the agent queries the LLM for N actions, showing it the
    last N observations (that it missed). Actions are returned one at a time,
    with new actions requested when the buffer is exhausted.
    
    Args:
        action_space: The environment's action space.
        observation_space: The environment's observation space (for context).
        n_actions: Number of steps/actions to request from the LLM (default 1).
        client: Optional LLMClient instance. If None, a new one is created
                with default settings.
        verbose: Whether to print debug info.
        device: Device hint (for compatibility with other agents).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        client:LLMClient,
        n_actions: int = 1,
        verbose: bool = False,
        device: str = "cpu",
    ):
        super().__init__(action_space, verbose=verbose)
        self.observation_space = observation_space
        self.n_actions = max(1, n_actions)
        self.device = device

        # Initialize LLM client and chat session
        self.client = client
        action_space_desc = self._describe_action_space()
        system_prompt = SYSTEM_PROMPT.format(action_space_description=action_space_desc)
        self.chat = ChatSession(
            self.client,
            system=system_prompt,
            max_messages=1,  # Keep only recent messages for independent calls
        )

        # Buffers for history and action queue
        self.observation_history: list[Any] = []
        self.action_history: list[Any] = []
        self.reward_history: list[float] = []
        self.action_queue: list[Any] = []
        self.step_count = 0
        
        # Persistent knowledge learned across episodes
        self.learned_knowledge = ""  # Accumulated knowledge string

        if self.verbose:
            print(f"[SimpleLLMAgent] Initialized with n_actions={self.n_actions}")

    def select_action(self, observation: Any) -> Any:
        """Select an action for the given observation.
        
        If the action queue is empty, queries the LLM for the next N actions.
        Otherwise, returns the next action from the queue.
        """
        self.observation_history.append(observation)

        # If we need more actions, query the LLM
        if not self.action_queue:
            self._query_llm_for_actions(observation)

        # Return the next action from the queue
        if self.action_queue:
            action = self.action_queue.pop(0)
        else:
            # Fallback: sample a random action if LLM parsing failed
            action = self.action_space.sample()
            if self.verbose:
                print(f"[SimpleLLMAgent] Warning: using fallback random action")

        self.action_history.append(action)
        self.step_count += 1
        return action

    def _query_llm_for_actions(self, current_observation: Any) -> None:
        """Query the LLM for the next N actions based on history."""
        if self.verbose:
            print(f"[SimpleLLMAgent] Querying LLM for {self.n_actions} actions "
                  f"(step {self.step_count})")

        # Format history for the prompt
        history_text = self._format_history()
        
        # Format current observation
        obs_str = self._format_observation(current_observation)
        
        # Format learned knowledge for context
        learned_knowledge_context = ""
        if self.learned_knowledge:
            learned_knowledge_context = f"Your current knowledge:\n{self.learned_knowledge}\n\n"

        # Create the prompt
        prompt = ACTION_REQUEST_PROMPT_TEMPLATE.format(
            learned_knowledge=learned_knowledge_context,
            n_actions=self.n_actions,
            history_text=history_text,
            current_observation=obs_str,
        )

        # Query the LLM
        try:
            response = self.chat.send(prompt)
            actions = self._parse_actions_from_response(response)
            self.action_queue.extend(actions)
            
            if self.verbose:
                print(f"[SimpleLLMAgent] Got {len(actions)} action(s): {actions}")
        except Exception as e:
            if self.verbose:
                print(f"[SimpleLLMAgent] Error querying LLM: {e}")
            # On error, the queue remains empty and we'll use fallback in select_action

    def _format_history(self) -> str:
        """Format the last N observations and actions for the prompt."""
        if not self.observation_history:
            return "No history yet."

        lines = []
        # Show the last n_actions observations
        recent_obs = self.observation_history[-self.n_actions:]
        for i, obs in enumerate(recent_obs):
            obs_str = self._format_observation(obs)
            step_num = self.step_count - len(recent_obs) + i
            lines.append(f"Step {step_num}: Observation = {obs_str}")
            # Show action if we took one
            action_idx = len(self.action_history) - len(recent_obs) + i
            if action_idx >= 0 and action_idx < len(self.action_history):
                action = self.action_history[action_idx]
                lines.append(f"           Action = {action}")

        return "\n".join(lines) if lines else "No history yet."

    def _format_observation(self, obs: Any) -> str:
        """Format an observation for inclusion in the prompt."""
        if isinstance(obs, (list, tuple)):
            return str(list(obs)[:10])  # Truncate long lists
        elif isinstance(obs, dict):
            return str({k: v for i, (k, v) in enumerate(obs.items()) if i < 5})
        else:
            return str(obs)

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

    def _parse_actions_from_response(self, response: str) -> list[Any]:
        """Parse actions from the LLM response.
        
        Expects a JSON response with format: {"actions": [action1, action2, ...]}
        """
        # Try to extract JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response[:200]}")

        try:
            data = json.loads(json_match.group())
            if "actions" not in data:
                raise ValueError("Response JSON missing 'actions' key")
            
            actions = data["actions"]
            if not isinstance(actions, list):
                actions = [actions]
            
            # Ensure we got the requested number of actions
            if len(actions) < self.n_actions:
                if self.verbose:
                    print(f"[SimpleLLMAgent] Warning: got {len(actions)} actions, "
                          f"expected {self.n_actions}")
            
            return actions[:self.n_actions]
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response[:300]}")

    def update(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        """Receive feedback after an action and generate episode summary when done."""
        # Track the reward for this step
        self.reward_history.append(reward)
        
        # When episode is done, update knowledge
        if done:
            self._update_learned_knowledge()

    def _update_learned_knowledge(self) -> None:
        """Update learned knowledge based on the episode trajectory."""
        if not self.observation_history or not self.action_history:
            return
        
        if self.verbose:
            print(f"[SimpleLLMAgent] Updating knowledge for episode "
                  f"(steps={len(self.action_history)}, reward={sum(self.reward_history):.2f})")
        
        # Format trajectory data
        observations_str = "\n".join(
            f"  Step {i}: {self._format_observation(obs)}"
            for i, obs in enumerate(self.observation_history)
        )
        actions_str = "\n".join(
            f"  Step {i}: {action}"
            for i, action in enumerate(self.action_history)
        )
        rewards_str = "\n".join(
            f"  Step {i}: {reward:.2f}"
            for i, reward in enumerate(self.reward_history)
        )
        total_return = sum(self.reward_history)
        
        # Query LLM to update knowledge base
        prompt = KNOWLEDGE_UPDATE_PROMPT_TEMPLATE.format(
            previous_knowledge=self.learned_knowledge if self.learned_knowledge else "No prior knowledge yet.",
            observations=observations_str,
            actions=actions_str,
            rewards=rewards_str,
            total_return=f"{total_return:.2f}",
        )
        
        try:
            updated_knowledge = self.chat.send(prompt)
            self.learned_knowledge = updated_knowledge
            
            if self.verbose:
                print(f"[SimpleLLMAgent] Knowledge updated: {updated_knowledge[:150]}...")
        except Exception as e:
            if self.verbose:
                print(f"[SimpleLLMAgent] Error updating knowledge: {e}")

    def reset(self) -> None:
        """Reset episode-specific state."""
        self.observation_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.action_queue.clear()
        self.step_count = 0
        # Note: We keep learned_knowledge across episodes to accumulate knowledge

        if self.verbose:
            print(f"[SimpleLLMAgent] Episode reset")

    def get_episode_data(self) -> dict[str, Any]:
        """Return episode data for logging."""
        return {
            "metrics": {
                "total_steps": self.step_count,
                "llm_calls": (self.step_count + self.n_actions - 1) // self.n_actions,
                "episode_return": sum(self.reward_history),
            },
            "artifacts": {
                "learned_knowledge.txt": self.learned_knowledge if self.learned_knowledge else "No knowledge learned yet.",
            },
        }
