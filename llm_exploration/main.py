from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt

from config import make_agent, make_env, ENV_CONFIGS, AGENT_CONFIGS
from training import train


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def make_config_folder_name(config: dict, resolved_params: dict) -> str:
    """Build a short, readable folder name from a config's ``folder_keys``.

    Generic by construction: it only ever looks at ``config["folder_keys"]``
    and the matching entries in ``resolved_params``, so new environments or
    agents never require touching this function.
    """
    folder_keys = config.get("folder_keys") or []

    if not folder_keys:
        return "default"

    parts = [
        f"{key.replace('_', '')}{resolved_params[key]}"
        for key in folder_keys
    ]

    return _safe_name("_".join(parts))


def run_multi_seed_experiment(
    env_fn,
    agent_fn,
    seeds: list[int],
    max_steps: int | None = None,
    num_episodes: int | None = None,
    tag: str = "default",
    cli_args: argparse.Namespace | None = None,
) -> Path:
    """Run multiple experiments with different random seeds."""

    outputs_dir = Path(__file__).resolve().parent / "outputs"

    # Temporary instances only to determine names.
    sample_env, sample_env_params = env_fn()
    sample_agent, sample_agent_params = agent_fn(sample_env)

    env_name = type(sample_env).__name__
    agent_name = type(sample_agent).__name__

    env_config_name = make_config_folder_name(ENV_CONFIGS[env_name], sample_env_params)
    agent_config_name = make_config_folder_name(AGENT_CONFIGS[agent_name], sample_agent_params)

    parent_dir = (
        outputs_dir
        / _safe_name(env_name)
        / env_config_name
        / _safe_name(agent_name)
        / agent_config_name
        / _safe_name(tag)
    )
    parent_dir.mkdir(parents=True, exist_ok=True)
   
    shutil.copy2(
        Path(__file__).resolve().parent / "config.py",
        parent_dir / "config.py",
    )

    (parent_dir / "config.json").write_text(
        json.dumps(
            {
                "environment": {"name": env_name, "params": sample_env_params},
                "agent": {"name": agent_name, "params": sample_agent_params},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if cli_args is not None:
        (parent_dir / "run_args.json").write_text(
            json.dumps(vars(cli_args), indent=2) + "\n",
            encoding="utf-8",
        )

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

    for seed_idx, seed, seed_dir in seeds_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running seed {seed_idx + 1}/{len(seeds)} (seed={seed})")
        print(f"{'=' * 60}\n")

        env, _ = env_fn()
        agent, _ = agent_fn(env)

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

    # Performance vs environment steps.
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

    # Performance vs LLM tokens.
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

    # Performance vs wall-clock time.
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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="SimpleGridEnv")
    parser.add_argument("--agent", default="HybridLLMDQNAgent")
    parser.add_argument("--llm", default="GEMINI")

    parser.add_argument("--env-overrides", default="{}")
    parser.add_argument("--agent-overrides", default="{}")

    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 45, 68])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--tag", default="default")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    env_overrides = json.loads(args.env_overrides)
    agent_overrides = json.loads(args.agent_overrides)

    run_multi_seed_experiment(
        env_fn=partial(
            make_env,
            args.env,
            overrides=env_overrides,
        ),
        agent_fn=partial(
            make_agent,
            agent_name=args.agent,
            llm_name=args.llm,
            overrides=agent_overrides,
        ),
        seeds=args.seeds,
        max_steps=args.max_steps,
        num_episodes=args.num_episodes,
        tag=args.tag,
        cli_args=args,
    )