from __future__ import annotations

import matplotlib.pyplot as plt

from agent import SimpleAgent, QLearningAgent, LLMAgent, HybridAgent
from env import HiddenChainEnv, HiddenRuleGridEnv


def run_agent_env_loop(max_steps: int = 20) -> list[tuple[int, float]]:
    # env = HiddenChainEnv()
    env = HiddenRuleGridEnv()
    
    # agent = QLearningAgent(env.action_space)
    agent = HybridAgent(env.action_space)
    episode_returns: list[tuple[int, float]] = []

    step_counter = 0
    episode_return = 0.0

    observation, _ = env.reset()
    action = agent.act(observation)
    print(env.render())

    while step_counter < max_steps:
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_counter += 1
        episode_return += float(reward)

        agent.update(observation, action, reward, next_observation, done)
        print(f"step={step_counter} action={action} reward={reward} {env.render()}")

        if done:
            episode_returns.append((step_counter, episode_return))
            observation, _ = env.reset()
            action = agent.act(observation)
            episode_return = 0.0
            continue

        observation = next_observation
        action = agent.act(observation)

    if episode_return:
        episode_returns.append((step_counter, episode_return))

    return episode_returns


def plot_rewards(episode_returns: list[tuple[int, float]]) -> None:
    if not episode_returns:
        return
    steps, returns = zip(*episode_returns)
    plt.plot(steps, returns)
    plt.xlabel("Step")
    plt.ylabel("Episode return")
    plt.title("Episode return at episode end")
    plt.show()


if __name__ == "__main__":
    # rewards = run_agent_env_loop(max_steps=200_000)
    rewards = run_agent_env_loop(max_steps=500)
    plot_rewards(rewards)
