from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _write_episode_artifacts(
    experiment_dir: Path,
    episode: int,
    episode_data: dict[str, Any],
) -> dict[str, Any]:
    """Write generic agent data and artifacts below one experiment directory."""
    metrics = episode_data.get("metrics", {})
    artifacts = episode_data.get("artifacts", {})

    if not isinstance(metrics, dict):
        raise TypeError("agent episode data 'metrics' must be a dictionary")
    if not isinstance(artifacts, dict):
        raise TypeError("agent episode data 'artifacts' must be a dictionary")

    # Keep the agent-specific JSON separate from the common training metrics.
    agent_data_dir = experiment_dir / "agent_data"
    agent_data_dir.mkdir(parents=True, exist_ok=True)

    agent_data_path = agent_data_dir / f"episode_{episode:04d}.json"
    agent_data_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    for relative_path, content in artifacts.items():
        relative_path = Path(relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Agent artifact path must stay inside the experiment directory: "
                f"{relative_path}"
            )

        # Give each episode's artifact its own file: "name.ext" -> "name_episode_0000.ext",
        # nested under agent_data/ alongside the per-episode metrics.
        stem, suffix = relative_path.stem, relative_path.suffix
        numbered_name = f"{stem}_episode_{episode:04d}{suffix}"
        path = agent_data_dir / relative_path.parent / numbered_name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(str(content), encoding="utf-8")

    return metrics


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def train(
    env,
    agent,
    num_episodes: int | None = None,
    num_steps: int | None = None,
    render: bool = False,
    verbose: bool = False,
    output_dir: str | Path | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Run the ordinary agent-environment interaction loop.

    Training stops when either ``num_episodes`` or the global ``num_steps``
    limit is reached. Either limit can be ``None``, but not both. If the step
    limit ends an episode, that partial episode is included in the results.

    The trainer knows only the Gymnasium API and the small BaseAgent API. It
    returns one dictionary per episode so experiments can easily add fields or
    turn the results into a DataFrame later.
    
    Args:
        env: Gymnasium environment.
        agent: Agent implementing BaseAgent API.
        num_episodes: Number of episodes to run.
        num_steps: Total environment steps limit.
        render: Whether to render episodes.
        verbose: Whether to print debug info.
        output_dir: Directory to save results.
        seed: Random seed for reproducibility (sets random, numpy, torch seeds).
    """
    
    # Set random seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
    
    if num_episodes is None and num_steps is None:
        raise ValueError("num_episodes and num_steps cannot both be None")
    if num_episodes is not None and num_episodes < 0:
        raise ValueError("num_episodes must be non-negative")
    if num_steps is not None and num_steps < 0:
        raise ValueError("num_steps must be non-negative")

    experiment_dir = Path(output_dir) if output_dir is not None else None
    metrics_path = None
    if experiment_dir is not None:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = experiment_dir / "metrics.jsonl"

    results = []
    total_steps = 0
    episode = 0
    training_start = time.perf_counter()

    while True:
        if num_episodes is not None and episode >= num_episodes:
            break
        if num_steps is not None and total_steps >= num_steps:
            break

        observation, _ = env.reset()
        agent.reset()

        episode_start = time.perf_counter()
        usage_before = agent.get_llm_usage()
        episode_return = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)

            steps += 1
            total_steps += 1
            episode_return += float(reward)
            done = terminated or truncated

            if num_steps is not None and total_steps >= num_steps:
                done = True

            agent.update(observation, action, reward, next_observation, done)

            if render:
                print(env.render())

            observation = next_observation

        # ``agent.update`` runs before this snapshot so any agent-side work
        # performed at the end of an episode is included in the measurements.
        usage_after = agent.get_llm_usage()
        episode_wall_time = time.perf_counter() - episode_start
        wall_time = time.perf_counter() - training_start

        prompt_tokens = max(
            0,
            int(usage_after.get("prompt_tokens", 0))
            - int(usage_before.get("prompt_tokens", 0)),
        )
        completion_tokens = max(
            0,
            int(usage_after.get("completion_tokens", 0))
            - int(usage_before.get("completion_tokens", 0)),
        )
        llm_tokens = max(
            0,
            int(usage_after.get("total_tokens", 0))
            - int(usage_before.get("total_tokens", 0)),
        )

        episode_data = agent.get_episode_data()
        if not isinstance(episode_data, dict):
            raise TypeError("agent.get_episode_data() must return a dictionary")

        agent_metrics = {}
        if experiment_dir is not None:
            agent_metrics = _write_episode_artifacts(
                experiment_dir,
                episode,
                episode_data,
            )
        else:
            agent_metrics = episode_data.get("metrics", {})

        episode_result = {
            "episode": episode,
            "return": episode_return,
            "steps": steps,
            "episode_steps": steps,
            "environment_steps": total_steps,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_tokens": llm_tokens,
            "cumulative_prompt_tokens": int(usage_after.get("prompt_tokens", 0)),
            "cumulative_completion_tokens": int(
                usage_after.get("completion_tokens", 0)
            ),
            "cumulative_llm_tokens": int(usage_after.get("total_tokens", 0)),
            "episode_wall_time": episode_wall_time,
            "wall_time": wall_time,
            "agent_data": agent_metrics,
        }
        results.append(episode_result)

        if metrics_path is not None:
            _append_jsonl(metrics_path, episode_result)

        if verbose:
            print(
                f"episode={episode} totalstep={total_steps} action={action} "
                f"episode_return={episode_return} done={done}"
            )
            
        episode += 1

    return results
