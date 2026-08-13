"""Compare multiple experiment runs on a single figure with 4 subplots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(data: list, window_size: int = 5) -> list:
    """Compute moving average of data to smooth plots.
    
    Args:
        data: Input data list.
        window_size: Size of the moving window. Defaults to 5.
    
    Returns:
        Smoothed data with the same length as input (padded at edges).
    """
    if len(data) < window_size:
        return data
    
    # Use numpy's convolve for efficient moving average
    kernel = np.ones(window_size) / window_size
    # 'same' mode pads the edges to maintain length
    smoothed = np.convolve(data, kernel, mode='same')
    return smoothed.tolist()


def load_experiment(experiment_dir: Path) -> dict:
    """Load results and metadata from an experiment directory.

    Supports both the old layout (results.json directly in experiment_dir)
    and the new per-seed layout (experiment_dir/seed_XXX/results.json),
    in which case results from all seeds are loaded for averaging.
    """
    metadata_file = experiment_dir / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    seed_dirs = sorted(experiment_dir.glob("seed_*"))
    if seed_dirs:
        seed_results = []
        for seed_dir in seed_dirs:
            results_file = seed_dir / "results.json"
            if not results_file.exists():
                continue
            with open(results_file) as f:
                seed_results.append(json.load(f))
        if not seed_results:
            raise FileNotFoundError(f"No seed results.json found in {experiment_dir}")
        return {"seed_results": seed_results, "metadata": metadata, "path": experiment_dir}

    results_file = experiment_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"No results.json in {experiment_dir}")

    with open(results_file) as f:
        results = json.load(f)

    return {"seed_results": [results], "metadata": metadata, "path": experiment_dir}


def average_over_seeds(seed_results: list, x_key: str, y_key: str = "return", num_points: int = 200):
    """Average a metric across seeds by interpolating each seed onto a common x grid."""
    xs = [np.array([r[x_key] for r in res]) for res in seed_results]
    ys = [np.array([r[y_key] for r in res]) for res in seed_results]

    x_max = min(x[-1] for x in xs)
    grid = np.linspace(0, x_max, num_points)

    interp_ys = [np.interp(grid, x, y) for x, y in zip(xs, ys)]
    avg_y = np.mean(interp_ys, axis=0)
    return grid, avg_y


def get_experiment_label(experiment_dir: Path) -> str:
    """Extract a clean label from the experiment directory name."""
    # Format: SimpleGridEnv_SimpleLLMAgent_20260812_161619_661425
    name = experiment_dir.name
    parts = name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return name


def plot_experiments(experiments_dict: dict[str, Path], output_file: Path | None = None, show: bool = True) -> None:
    """Plot performance metrics for multiple experiments on 4 axes.
    
    Args:
        experiments_dict: Dictionary mapping legend labels to experiment directory paths.
        output_file: Optional path to save the figure.
        show: Whether to display the figure.
    """
    
    experiments = []
    for label, exp_dir in experiments_dict.items():
        try:
            exp_data = load_experiment(Path(exp_dir))
            exp_data["label"] = label  # Use provided label
            experiments.append(exp_data)
            print(f"Loaded: {label} ({exp_dir})")
        except Exception as e:
            print(f"Failed to load {exp_dir}: {e}")
    
    if not experiments:
        print("No experiments loaded.")
        return
    
    # Create figure with 4 subplots: env steps, prompt tokens, completion tokens, wall time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance Comparison Across Experiments", fontsize=16, fontweight="bold")
    
    ax_env_steps = axes[0, 0]
    ax_prompt_tokens = axes[0, 1]
    ax_completion_tokens = axes[1, 0]
    ax_wall_time = axes[1, 1]
    
    # Plot each experiment (averaged over seeds, then smoothed)
    for exp_data in experiments:
        seed_results = exp_data["seed_results"]
        label = exp_data["label"]  # Use the provided label from dictionary

        # Axis 1: Environment steps
        env_steps, returns_avg = average_over_seeds(seed_results, "environment_steps")
        ax_env_steps.plot(
            env_steps, moving_average(returns_avg.tolist(), window_size=5),
            marker="o", markersize=3, linewidth=1.5, label=label, alpha=0.8
        )

        # Axis 2: Prompt tokens
        prompt_tokens, returns_avg = average_over_seeds(seed_results, "cumulative_prompt_tokens")
        ax_prompt_tokens.plot(
            prompt_tokens, moving_average(returns_avg.tolist(), window_size=5),
            marker="o", markersize=3, linewidth=1.5, label=label, alpha=0.8
        )

        # Axis 3: Completion tokens
        completion_tokens, returns_avg = average_over_seeds(seed_results, "cumulative_completion_tokens")
        ax_completion_tokens.plot(
            completion_tokens, moving_average(returns_avg.tolist(), window_size=5),
            marker="o", markersize=3, linewidth=1.5, label=label, alpha=0.8
        )

        # Axis 4: Wall time
        wall_time, returns_avg = average_over_seeds(seed_results, "wall_time")
        ax_wall_time.plot(
            wall_time, moving_average(returns_avg.tolist(), window_size=5),
            marker="o", markersize=3, linewidth=1.5, label=label, alpha=0.8
        )
    
    # Format axes
    ax_env_steps.set_xlabel("Cumulative Environment Steps")
    ax_env_steps.set_ylabel("Episode Return")
    ax_env_steps.set_title("Return vs Environment Steps")
    ax_env_steps.grid(alpha=0.25)
    ax_env_steps.legend(fontsize=9)
    
    ax_prompt_tokens.set_xlabel("Cumulative Prompt Tokens")
    ax_prompt_tokens.set_ylabel("Episode Return")
    ax_prompt_tokens.set_title("Return vs Prompt Tokens")
    ax_prompt_tokens.grid(alpha=0.25)
    ax_prompt_tokens.legend(fontsize=9)
    
    ax_completion_tokens.set_xlabel("Cumulative Completion Tokens")
    ax_completion_tokens.set_ylabel("Episode Return")
    ax_completion_tokens.set_title("Return vs Completion Tokens")
    ax_completion_tokens.grid(alpha=0.25)
    ax_completion_tokens.legend(fontsize=9)
    
    ax_wall_time.set_xlabel("Elapsed Wall-Clock Time (seconds)")
    ax_wall_time.set_ylabel("Episode Return")
    ax_wall_time.set_title("Return vs Wall-Clock Time")
    ax_wall_time.grid(alpha=0.25)
    ax_wall_time.legend(fontsize=9)
    
    fig.tight_layout()
    
    # Save if output file specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    """Main entry point."""
    # Define experiments to plot: label -> experiment directory path
    experiments_dict = {
        # "DQN": "outputs/SimpleGridEnv_DQNAgent_baseline",
        # "LLM-Agent": "outputs/SimpleGridEnv_SimpleLLMAgent_baseline",
        # "Hybrid (10% LLM)": "outputs/SimpleGridEnv_HybridLLMDQNAgent_freq_10pct",
        
        "LLM-Agent-50": "outputs/SimpleGridEnv_SimpleLLMAgent_test_50",
        "LLM-Agent-20": "outputs/SimpleGridEnv_SimpleLLMAgent_test",
        "Programmatic-50": "outputs/SimpleGridEnv_ProgrammaticLLMAgent_test_50", 
    }
    
    # Optional: set output file to save the plot
    output_file = Path("comparison2.png")  # Set to Path("comparison.png") to save
    
    plot_experiments(experiments_dict, output_file=output_file, show=True)


if __name__ == "__main__":
    main()
