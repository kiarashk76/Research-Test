"""HybridLLMDQNAgent: Combines DQNAgent and SimpleLLMAgent for collaborative learning."""

from __future__ import annotations

import random
from typing import Any, Optional

from llm import LLMClient

from .base import BaseAgent
from .dqn import DQNAgent
from .simple_llm import SimpleLLMAgent


class HybridLLMDQNAgent(BaseAgent):
    """Hybrid agent that combines DQN and SimpleLLMAgent.
    
    Each episode, one of the two agents is selected to play based on llm_freq probability.
    Regardless of which plays, both agents learn from all transitions:
    - DQN learns by adding all transitions to its replay buffer
    - SimpleLLM learns by incorporating all trajectories into its knowledge base
    
    This creates a curriculum where agents learn from each other's experiences.
    
    Args:
        observation_space: Environment observation space.
        action_space: Environment action space.
        llm_freq: Probability (0.0-1.0) that SimpleLLMAgent plays an episode.
                 Default 0.5 means roughly 50% episodes are LLM, 50% DQN.
        client: LLMClient instance for SimpleLLMAgent.
        dqn_kwargs: Keyword arguments to pass to DQNAgent.
        llm_kwargs: Keyword arguments to pass to SimpleLLMAgent.
        verbose: Whether to print debug info.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        llm_freq: float = 0.5,
        client: Optional[LLMClient] = None,
        dqn_kwargs: dict | None = None,
        llm_kwargs: dict | None = None,
        verbose: bool = False,
    ):
        super().__init__(action_space, verbose=verbose)
        
        # Validate llm_freq
        if not 0.0 <= llm_freq <= 1.0:
            raise ValueError("llm_freq must be between 0.0 and 1.0")
        
        self.llm_freq = llm_freq
        self.observation_space = observation_space
        
        # Initialize DQN agent
        dqn_kwargs = dqn_kwargs or {}
        self.dqn_agent = DQNAgent(
            observation_space=observation_space,
            action_space=action_space,
            verbose=verbose,
            **dqn_kwargs,
        )
        
        # Initialize SimpleLLM agent
        llm_kwargs = llm_kwargs or {}
        self.llm_agent = SimpleLLMAgent(
            action_space=action_space,
            observation_space=observation_space,
            client=client,
            verbose=verbose,
            **llm_kwargs,
        )
        
        # Track which agent is playing this episode
        self.use_llm_this_episode = False
        self.episode_count = 0
        self.episode_started = False
        
        if self.verbose:
            print(f"[HybridLLMDQNAgent] Initialized with llm_freq={llm_freq}")

    def select_action(self, observation: Any) -> Any:
        """Select action using the agent designated for this episode."""
        if self.use_llm_this_episode:
            return self.llm_agent.select_action(observation)
        else:
            return self.dqn_agent.select_action(observation)

    def update(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        """Update both agents with the transition data.
        
        Both agents learn from all transitions regardless of who played:
        - DQN adds to replay buffer and trains
        - SimpleLLM updates knowledge at episode end
        """
        # Always update DQN with the transition
        self.dqn_agent.update(observation, action, reward, next_observation, done)
        
        # Always update SimpleLLM with the transition
        self.llm_agent.update(observation, action, reward, next_observation, done)
        
        # When episode ends, increment counter
        if done:
            self.episode_count += 1

    def reset(self) -> None:
        """Reset both agents for a new episode and decide which agent plays."""
        self.dqn_agent.reset()
        self.llm_agent.reset()
        
        # Decide which agent plays this episode
        self.use_llm_this_episode = random.random() < self.llm_freq
        self.episode_started = True
        
        if self.verbose:
            agent_name = "LLM" if self.use_llm_this_episode else "DQN"
            print(f"[HybridLLMDQNAgent] Episode {self.episode_count}: Using {agent_name}")

    def get_llm_usage(self) -> dict[str, int]:
        """Return combined LLM token usage from SimpleLLMAgent."""
        return self.llm_agent.get_llm_usage()

    def get_episode_data(self) -> dict[str, Any]:
        """Return combined episode data from both agents."""
        dqn_data = self.dqn_agent.get_episode_data()
        llm_data = self.llm_agent.get_episode_data()
        
        return {
            "metrics": {
                "total_steps": self.dqn_agent.total_steps,
                "episode_count": self.episode_count,
                "llm_episodes": sum(
                    1 for _ in range(self.episode_count)
                    if random.random() < self.llm_freq
                ),  # Approximate
                "dqn_replay_buffer_size": len(self.dqn_agent.replay_buffer),
                "dqn_metrics": dqn_data.get("metrics", {}),
                "llm_metrics": llm_data.get("metrics", {}),
            },
            "artifacts": {
                "dqn_data.txt": str(dqn_data.get("artifacts", {})),
                "llm_learned_knowledge.txt": llm_data.get("artifacts", {}).get("learned_knowledge.txt", ""),
            },
        }
