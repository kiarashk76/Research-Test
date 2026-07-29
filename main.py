from __future__ import annotations

import argparse
from pathlib import Path

from experiment import ExperimentConfig, run_experiment
from utils import mean_final_mse


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run the value-prediction plasticity experiment.")
    parser.add_argument("--num-states", type=int, default=100)
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--steps-per-task", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("results/value_prediction_plasticity"))
    parser.add_argument(
        "--reset-continual-optimizer-per-task",
        action="store_true",
        help="Reset optimizer state for the continual network at each task boundary.",
    )
    args = parser.parse_args()

    return ExperimentConfig(
        num_states=args.num_states,
        num_tasks=args.num_tasks,
        steps_per_task=args.steps_per_task,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        activation=args.activation,
        num_seeds=args.num_seeds,
        seed_offset=args.seed_offset,
        device=args.device,
        output_dir=args.output_dir,
        reset_continual_optimizer_per_task=args.reset_continual_optimizer_per_task,
    )


def main() -> None:
    config = parse_args()
    rows = run_experiment(config)

    final_task = config.num_tasks
    continual = mean_final_mse(rows, final_task, "continual_final_mse")
    fresh = mean_final_mse(rows, final_task, "fresh_final_mse")

    print(f"Wrote CSV: {config.output_dir / 'plasticity_results.csv'}")
    print(f"Wrote plot: {config.output_dir / 'mse_after_500_updates.png'}")
    print(f"Final task mean continual MSE: {continual:.6g}")
    print(f"Final task mean fresh MSE: {fresh:.6g}")
    print(f"Final task continual/fresh ratio: {continual / max(fresh, 1e-12):.6g}")


if __name__ == "__main__":
    main()
