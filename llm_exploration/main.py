from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from agents import *
from environments import *
from llm.client import LLMClient
from training import train


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def _create_experiment_dir(env, agent) -> Path:
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = "_".join(
        (
            _safe_name(type(env).__name__),
            _safe_name(type(agent).__name__),
            timestamp,
        )
    )
    experiment_dir = outputs_dir / name
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir

def run_multi_seed_experiment(
    env_fn,
    agent_fn,
    seeds: list[int] | None = None,
    max_steps: int | None = None,
    num_episodes: int | None = None,
    tag: str = "default",
) -> Path:
    """Run multiple experiments with different random seeds."""

    if seeds is None:
        seeds = [42, 43, 44]

    outputs_dir = Path(__file__).resolve().parent / "outputs"

    # Create temporary instances only to automatically determine names
    sample_env = env_fn()
    sample_agent = agent_fn(sample_env)

    env_name = type(sample_env).__name__
    agent_name = type(sample_agent).__name__

    experiment_name = "_".join(
        [
            _safe_name(env_name),
            _safe_name(agent_name),
            _safe_name(tag),
        ]
    )

    parent_dir = outputs_dir / experiment_name
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Determine which seeds still need to run
    seeds_to_run = []
    seeds_skipped = []

    for seed_idx, seed in enumerate(seeds):
        seed_dir = parent_dir / f"seed_{seed:03d}"
        results_file = seed_dir / "results.json"

        if results_file.exists():
            seeds_skipped.append(seed)
            print(f"✓ Seed {seed} already finished (skipping)")
        else:
            seeds_to_run.append((seed_idx, seed, seed_dir))

    all_results = {}

    for seed_idx, seed, seed_dir in seeds_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running seed {seed_idx + 1}/{len(seeds)} (seed={seed})")
        print(f"{'=' * 60}\n")

        # Fresh environment and agent for every seed
        env = env_fn()
        agent = agent_fn(env)

        seed_dir.mkdir(parents=True, exist_ok=True)

        results = train(
            env,
            agent,
            num_episodes=num_episodes,
            num_steps=max_steps,
            render=False,
            verbose=True,
            output_dir=seed_dir,
            seed=seed,
        )

        (seed_dir / "results.json").write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )

        metadata = {
            "environment": type(env).__name__,
            "agent": type(agent).__name__,
            "seed": seed,
            "tag": tag,
            "created_at": datetime.now().isoformat(),
            "max_steps": max_steps,
            "num_episodes": num_episodes,
        }

        (seed_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        all_results[seed] = results

        plot_training_metrics(
            results,
            seed_dir / "plots",
        )

    agg_metadata = {
        "environment": env_name,
        "agent": agent_name,
        "seeds": seeds,
        "tag": tag,
        "seeds_completed": sorted(
            seeds_skipped + [seed for _, seed, _ in seeds_to_run]
        ),
        "seeds_skipped": seeds_skipped,
        "created_at": datetime.now().isoformat(),
        "max_steps": max_steps,
        "num_episodes": num_episodes,
    }

    (parent_dir / "metadata.json").write_text(
        json.dumps(agg_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print("Experiment summary:")
    print(f"  Environment: {env_name}")
    print(f"  Agent: {agent_name}")
    print(f"  New seeds run: {len(seeds_to_run)}")
    print(f"  Seeds skipped: {len(seeds_skipped)}")
    print(f"  Results saved to: {parent_dir}")
    print(f"{'=' * 60}\n")

    return parent_dir

def plot_training_metrics(
    episode_metrics: list[dict],
    output_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """Save performance curves against steps, tokens, and wall-clock time."""
    if not episode_metrics:
        return

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    returns = [result["return"] for result in episode_metrics]

    def save_figure(figure, filename: str) -> None:
        if output_path is not None:
            figure.savefig(
                output_path / filename,
                dpi=150,
                bbox_inches="tight",
            )
        if not show:
            plt.close(figure)

    # Plot 1: performance versus environment interaction cost.
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [result["environment_steps"] for result in episode_metrics],
        returns,
        marker="o",
        markersize=2,
        linewidth=0.8,
    )
    axis.set_xlabel("Cumulative environment steps")
    axis.set_ylabel("Episode return")
    axis.set_title("Episode return vs environment steps")
    axis.grid(alpha=0.25)
    save_figure(figure, "return_vs_environment_steps.png")

    # Plot 2: prompt and completion token budgets in separate subplots.
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].plot(
        [result["cumulative_prompt_tokens"] for result in episode_metrics],
        returns,
        marker="o",
        markersize=2,
        linewidth=0.8,
    )
    axes[0].set_xlabel("Cumulative prompt tokens")
    axes[0].set_ylabel("Episode return")
    axes[0].set_title("Prompt tokens")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        [result["cumulative_completion_tokens"] for result in episode_metrics],
        returns,
        marker="o",
        markersize=2,
        linewidth=0.8,
    )
    axes[1].set_xlabel("Cumulative completion tokens")
    axes[1].set_title("Completion tokens")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    save_figure(figure, "return_vs_llm_tokens.png")

    # Plot 3: performance versus elapsed monotonic wall-clock time.
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [result["wall_time"] for result in episode_metrics],
        returns,
        marker="o",
        markersize=2,
        linewidth=0.8,
    )
    axis.set_xlabel("Elapsed wall-clock time (seconds)")
    axis.set_ylabel("Episode return")
    axis.set_title("Episode return vs wall-clock time")
    axis.grid(alpha=0.25)
    save_figure(figure, "return_vs_wall_time.png")

    if show:
        plt.show()


def plot_rewards(episode_metrics: list[dict]) -> None:
    """Backward-compatible interactive wrapper for the training plots."""
    plot_training_metrics(episode_metrics, show=True)


if __name__ == "__main__":
    def make_env():
        return SimpleGridEnv(
            max_steps=50,
            size=5,
        )

    def make_agent(env):
        llm_name = "GEMINI" #VULCAN, GEMINI
        llm_client = LLMClient(
            model=os.environ.get(f"{llm_name}_MODEL"),
            api_key=os.environ.get(f"{llm_name}_API_KEY"),
            base_url=os.environ.get(f"{llm_name}_BASE_URL"),
            timeout=60,  # seconds
            max_retries=3,  # Number of retries for transient errors
            temperature=0.7,  # Creativity of responses
            stream=True,  # Stream responses for faster action retrieval  
        ) 
        # return DQNAgent(
        #     env.observation_space,
        #     env.action_space,
        #     verbose=False,
        #     device="cpu",
        #     num_batches=5,
        # )
        # return SimpleLLMAgent(
        #     env.observation_space,
        #     env.action_space,
        #     client=llm_client,
        #     n_actions=10,
        #     verbose=False,
        #     device="cpu",
        # )
        # return HybridLLMDQNAgent(
        #     env.observation_space,
        #     env.action_space,
        #     llm_freq=0.1,  # 10% of episodes use LLM
        #     client=llm_client,
        #     verbose=False,
        #     llm_kwargs={
        #         "n_actions": 10,
        #     },
        #     dqn_kwargs={
        #         "device": "cpu",
        #         "num_batches": 5,
        #     }
        # )
        return ProgrammaticLLMAgent(
            env.observation_space,
            env.action_space,
            client=llm_client,
            verbose=False,
            device="cpu",
            n_actions=50,
        )
        

    run_multi_seed_experiment(
        env_fn=make_env,
        agent_fn=make_agent,
        seeds=[23],#, 45, 68, 123, 456],
        max_steps=500,
        tag="test",
    )