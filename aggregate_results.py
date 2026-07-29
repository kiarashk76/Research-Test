from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment import ExperimentConfig, plot_results
from utils import read_results_csv, write_csv


def load_config(path: Path, output_dir: Path, num_seeds: int) -> ExperimentConfig:
    with path.open() as handle:
        data = json.load(handle)
    data["output_dir"] = output_dir
    data["num_seeds"] = num_seeds
    return ExperimentConfig(**data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-seed plasticity job-array outputs.")
    parser.add_argument("--input-root", type=Path, default=Path("results/value_prediction_plasticity"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/value_prediction_plasticity/combined"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_dirs = sorted(path for path in args.input_root.glob("seed_*") if path.is_dir())
    csv_paths = [path / "plasticity_results.csv" for path in seed_dirs]
    csv_paths = [path for path in csv_paths if path.exists()]

    if not csv_paths:
        raise SystemExit(f"No per-seed CSVs found under {args.input_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    for csv_path in csv_paths:
        rows.extend(read_results_csv(csv_path))

    rows.sort(key=lambda row: (int(row["task"]), int(row["seed"])))
    config = load_config(csv_paths[0].parent / "config.json", args.output_dir, len({int(row["seed"]) for row in rows}))

    write_csv(args.output_dir / "plasticity_results.csv", rows)
    plot_results(rows, config, args.output_dir / "mse_after_500_updates.png")

    print(f"Aggregated {len(csv_paths)} seed files")
    print(f"Wrote CSV: {args.output_dir / 'plasticity_results.csv'}")
    print(f"Wrote plot: {args.output_dir / 'mse_after_500_updates.png'}")


if __name__ == "__main__":
    main()
