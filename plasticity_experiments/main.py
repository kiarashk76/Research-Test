from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

from analyze import run_supervised_analysis
from experiments import run_rl_experiment, run_supervised_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run plasticity experiments.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--experiment", choices=["supervised", "rl"], default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def save_config(config: dict, output_path: Path) -> None:
    with output_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    if args.experiment is not None:
        base_config["experiment"] = args.experiment

    experiment_name = base_config["experiment"]
    intervention_type = base_config["intervention"]["type"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{intervention_type}_{timestamp}"
    run_dir = Path(base_config.get("output_dir", "results")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_config = deepcopy(base_config)
    saved_config["run_dir"] = str(run_dir)
    save_config(saved_config, run_dir / "config.yaml")

    results = []
    for seed in base_config["seeds"]:
        config = deepcopy(base_config)
        config["seed"] = int(seed)
        config["run_dir"] = str(run_dir)

        if experiment_name == "supervised":
            result = run_supervised_experiment(config)
        elif experiment_name == "rl":
            result = run_rl_experiment(config)
        else:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        results.append(result)
        print(f"Seed {seed}: saved {result['csv_path']}")
        if result.get("task_data_path"):
            print(f"Seed {seed}: saved task data {result['task_data_path']}")

    print(f"\nSaved run directory: {run_dir}")
    print(f"Saved {len(results)} seed CSV files.")
    if experiment_name == "supervised":
        print("\nRunning supervised analysis...")
        run_supervised_analysis([run_dir], run_dir / "analysis")


if __name__ == "__main__":
    main()
