from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from interventions import (
    reset_dormant_neurons,
    reset_random_neurons,
    reset_small_gradient_neurons,
)
from interventions.gradient_reset import current_gradient_scores
from rl_agents import DQNAgent
from rl_envs import SwitchingMDP
from .artifacts import save_eval_artifact


def run_rl_experiment(config: dict) -> dict:
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    rl_config = config["rl"]
    intervention_config = config["intervention"]
    intervention_type = intervention_config["type"]
    output_dir = Path(config["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(config, seed_offset=0)
    agent = make_agent(config, seed_offset=0)
    rows: list[dict] = []
    global_step = 0

    for task_index in range(rl_config["num_tasks"]):
        env.set_task(task_index)
        if intervention_type == "fresh":
            agent = make_agent(config, seed_offset=10_000 + task_index)

        activation_sums = None
        gradient_sums = None
        window_count = 0

        for step in range(1, rl_config["steps_per_task"] + 1):
            global_step += 1
            epsilon = linear_epsilon(
                step,
                rl_config["steps_per_task"],
                rl_config["epsilon_start"],
                rl_config["epsilon_end"],
            )

            td_loss = take_env_step(env, agent, epsilon, rl_config)
            activation_sums = update_activation_sums(agent, env, activation_sums)
            gradient_sums = update_gradient_sums(agent, gradient_sums)
            window_count += 1

            reset_count = 0
            if step % intervention_config["interval"] == 0:
                reset_count = apply_intervention(
                    agent,
                    intervention_config,
                    activation_sums,
                    gradient_sums,
                    window_count,
                )
                agent.update_target_network()
                activation_sums = None
                gradient_sums = None
                window_count = 0

            if step % rl_config["target_update_interval"] == 0:
                agent.update_target_network()

            if step % rl_config["eval_interval"] == 0 or step == rl_config["steps_per_task"]:
                eval_return = evaluate_agent(env, agent, rl_config["eval_episodes"])
                eval_inputs = torch.tensor(
                    env.observations,
                    dtype=torch.float32,
                    device=agent.device,
                )
                artifact_path = save_eval_artifact(
                    config=config,
                    run_dir=output_dir,
                    seed=seed,
                    actor=intervention_type,
                    network=agent.network,
                    eval_inputs=eval_inputs,
                    task_index=task_index,
                    step_within_task=step,
                    global_step=global_step,
                    metadata={
                        "experiment": "rl",
                        "eval_return": eval_return,
                        "td_loss": td_loss,
                        "epsilon": epsilon,
                        "intervention_type": intervention_type,
                        "num_neurons_reset": reset_count,
                    },
                    extra_state_dicts={"target_network": agent.target_network},
                )
                rows.append(
                    {
                        "seed": seed,
                        "task_index": task_index,
                        "step_within_task": step,
                        "global_step": global_step,
                        "agent": intervention_type,
                        "eval_return": eval_return,
                        "td_loss": td_loss,
                        "epsilon": epsilon,
                        "intervention_type": intervention_type,
                        "num_neurons_reset": reset_count,
                        "artifact_path": artifact_path,
                    }
                )

    csv_path = output_dir / f"seed_{seed}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"csv_path": str(csv_path), "num_rows": len(rows)}


def make_env(config: dict, seed_offset: int) -> SwitchingMDP:
    rl_config = config["rl"]
    return SwitchingMDP(
        num_states=rl_config["num_states"],
        obs_dim=rl_config["obs_dim"],
        num_actions=rl_config["num_actions"],
        num_tasks=rl_config["num_tasks"],
        seed=int(config["seed"]) + seed_offset,
    )


def make_agent(config: dict, seed_offset: int) -> DQNAgent:
    rl_config = config["rl"]
    network_config = config["network"]
    return DQNAgent(
        obs_dim=rl_config["obs_dim"],
        num_actions=rl_config["num_actions"],
        hidden_sizes=list(network_config["hidden_sizes"]),
        activation=network_config["activation"],
        learning_rate=rl_config["learning_rate"],
        replay_size=rl_config["replay_size"],
        batch_size=rl_config["batch_size"],
        gamma=rl_config["gamma"],
        seed=int(config["seed"]) + seed_offset,
    )


def take_env_step(env: SwitchingMDP, agent: DQNAgent, epsilon: float, rl_config: dict) -> float:
    state, _ = env.reset()
    action = agent.select_action(state, epsilon=epsilon, explore=True)
    next_state, reward, done, truncated, _ = env.step(action)
    agent.store_transition(state, action, reward, next_state, done or truncated)
    if len(agent.replay_buffer) < rl_config["update_after"]:
        return np.nan
    metrics = agent.update()
    return float(metrics["td_loss"])


def evaluate_agent(env: SwitchingMDP, agent: DQNAgent, episodes: int) -> float:
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        action = agent.select_action(state, explore=False)
        _, reward, _, _, _ = env.step(action)
        returns.append(reward)
    return float(np.mean(returns))


def linear_epsilon(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    fraction = min(1.0, step / total_steps)
    return start + fraction * (end - start)


def update_activation_sums(agent: DQNAgent, env: SwitchingMDP, activation_sums):
    observations = torch.tensor(env.observations, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        _, activations = agent.network(observations, return_activations=True)
    batch_scores = [activation.detach().abs().mean(dim=0) for activation in activations]
    if activation_sums is None:
        return [score.clone() for score in batch_scores]
    return [old + score for old, score in zip(activation_sums, batch_scores)]


def update_gradient_sums(agent: DQNAgent, gradient_sums):
    scores = current_gradient_scores(agent.network)
    if gradient_sums is None:
        return [score.clone() for score in scores]
    return [old + score for old, score in zip(gradient_sums, scores)]


def apply_intervention(
    agent: DQNAgent,
    intervention_config: dict,
    activation_sums,
    gradient_sums,
    window_count: int,
) -> int:
    intervention_type = intervention_config["type"]
    if intervention_type in {"none", "fresh"}:
        return 0
    if intervention_type == "random":
        info = reset_random_neurons(
            agent.network,
            intervention_config["reset_fraction"],
            optimizer=agent.optimizer,
        )
        return int(info["num_reset"])
    if intervention_type == "dormant":
        if activation_sums is None:
            return 0
        activation_means = [value / max(window_count, 1) for value in activation_sums]
        info = reset_dormant_neurons(
            agent.network,
            activation_means,
            intervention_config["dormant_threshold"],
            optimizer=agent.optimizer,
        )
        return int(info["num_reset"])
    if intervention_type == "small_gradient":
        if gradient_sums is None:
            return 0
        gradient_means = [value / max(window_count, 1) for value in gradient_sums]
        info = reset_small_gradient_neurons(
            agent.network,
            intervention_config["reset_fraction"],
            optimizer=agent.optimizer,
            gradient_scores=gradient_means,
        )
        return int(info["num_reset"])
    raise ValueError(f"Unknown intervention type: {intervention_type}")
