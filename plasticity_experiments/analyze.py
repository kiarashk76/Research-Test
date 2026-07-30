from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from analysis.metrics import (
    activation_erank_by_eval_step,
    dormant_neuron_fraction_by_eval_step,
    eval_mse_auc_by_task,
    eval_mse_by_eval_step,
    latest_eval_mse_by_task,
    prepare_supervised_eval_df,
    steps_to_mse_threshold_by_task,
)
from analysis.plots import (
    plot_activation_erank_by_eval_step,
    plot_dormant_neuron_fraction_by_eval_step,
    plot_eval_mse_auc_by_task,
    plot_eval_mse_by_eval_step,
    plot_latest_eval_mse_by_task,
    plot_steps_to_mse_threshold_by_task,
)


# Edit these when you want to run analysis without passing paths on the command line.
# Use one path for normal single-run analysis, or multiple paths for comparison.
RUN_DIRS = [
    "results/supervised_none_20260729_162251",
]
OUTPUT_DIR = None  # e.g. "results/my_comparison"; only used for multiple RUN_DIRS.
MSE_THRESHOLD = None  # If None, use analysis.steps_to_mse_threshold_by_task.mse_threshold.
PLOT_FLAGS = None  # If None, use analysis plot flags from config.yaml.

PLOT_NAMES = [
    "latest_eval_mse_by_task",
    "eval_mse_by_eval_step",
    "eval_mse_auc_by_task",
    "steps_to_mse_threshold_by_task",
    "activation_erank_by_eval_step",
    "dormant_neuron_fraction_by_eval_step",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze supervised plasticity experiment runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="One or more supervised run directories containing seed_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save multi-run comparison outputs. Defaults to <parent>/comparison_analysis.",
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=None,
        help="Eval MSE threshold for the steps-to-threshold plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir_values = args.run_dirs or RUN_DIRS
    if not run_dir_values:
        raise SystemExit(
            "No run directories provided. Either pass paths on the command line "
            "or edit RUN_DIRS near the top of analyze.py."
        )

    run_dirs = [Path(path) for path in run_dir_values]
    output_dir = analysis_output_dir(run_dirs, args.output_dir or OUTPUT_DIR)
    mse_threshold = args.mse_threshold
    if mse_threshold is None:
        mse_threshold = MSE_THRESHOLD
    run_supervised_analysis(run_dirs, output_dir, mse_threshold=mse_threshold)


def run_supervised_analysis(
    run_dirs: list[str | Path],
    output_dir: str | Path | None = None,
    mse_threshold: float | None = None,
    plot_flags: dict[str, bool] | None = None,
    clear_existing: bool = True,
) -> list[Path]:
    run_dirs = [Path(path) for path in run_dirs]
    if output_dir is None:
        output_dir = analysis_output_dir(run_dirs, None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        clear_previous_analysis_outputs(output_dir)

    df = pd.concat(
        [prepare_supervised_eval_df(run_dir) for run_dir in run_dirs],
        ignore_index=True,
    )
    if mse_threshold is None:
        mse_threshold = infer_mse_threshold(run_dirs)
    if plot_flags is None:
        plot_flags = infer_plot_flags(run_dirs)

    output_paths = []

    print(f"Analyzed {len(run_dirs)} supervised run(s).")
    print("Metrics are averaged across seeds; SEM is shown as the shaded band.")
    print(f"Analysis outputs: {output_dir}")

    if plot_flags["latest_eval_mse_by_task"]:
        latest = latest_eval_mse_by_task(df)
        latest_csv = output_dir / "latest_eval_mse_by_task.csv"
        latest_plot = output_dir / "latest_eval_mse_by_task.png"
        latest.to_csv(latest_csv, index=False)
        plot_latest_eval_mse_by_task(latest, latest_plot)
        output_paths.extend([latest_csv, latest_plot])
        print_table("Latest Eval MSE By Task", latest)

    if plot_flags["eval_mse_by_eval_step"]:
        eval_steps = eval_mse_by_eval_step(df)
        eval_steps_csv = output_dir / "eval_mse_by_eval_step.csv"
        eval_steps_plot = output_dir / "eval_mse_by_eval_step.png"
        eval_steps.to_csv(eval_steps_csv, index=False)
        plot_eval_mse_by_eval_step(eval_steps, eval_steps_plot)
        output_paths.extend([eval_steps_csv, eval_steps_plot])

    if plot_flags["eval_mse_auc_by_task"]:
        auc = eval_mse_auc_by_task(df)
        auc_csv = output_dir / "eval_mse_auc_by_task.csv"
        auc_plot = output_dir / "eval_mse_auc_by_task.png"
        auc.to_csv(auc_csv, index=False)
        plot_eval_mse_auc_by_task(auc, auc_plot)
        output_paths.extend([auc_csv, auc_plot])
        print_table("Eval MSE AUC By Task", auc)

    if plot_flags["steps_to_mse_threshold_by_task"]:
        threshold_steps = steps_to_mse_threshold_by_task(df, mse_threshold)
        threshold_steps_csv = output_dir / "steps_to_mse_threshold_by_task.csv"
        threshold_steps_plot = output_dir / "steps_to_mse_threshold_by_task.png"
        threshold_steps.to_csv(threshold_steps_csv, index=False)
        plot_steps_to_mse_threshold_by_task(
            threshold_steps,
            threshold_steps_plot,
            mse_threshold=mse_threshold,
        )
        output_paths.extend([threshold_steps_csv, threshold_steps_plot])
        print(f"Steps-to-threshold uses Eval MSE <= {mse_threshold:g}.")
        print_table("Steps To MSE Threshold By Task", threshold_steps)

    if plot_flags["activation_erank_by_eval_step"]:
        try:
            erank = activation_erank_by_eval_step(df)
        except ValueError as error:
            print(f"Skipping activation_erank_by_eval_step: {error}")
        else:
            erank_csv = output_dir / "activation_erank_by_eval_step.csv"
            erank_plot = output_dir / "activation_erank_by_eval_step.png"
            erank.to_csv(erank_csv, index=False)
            plot_activation_erank_by_eval_step(erank, erank_plot)
            output_paths.extend([erank_csv, erank_plot])

    if plot_flags["dormant_neuron_fraction_by_eval_step"]:
        try:
            dormant = dormant_neuron_fraction_by_eval_step(df)
        except ValueError as error:
            print(f"Skipping dormant_neuron_fraction_by_eval_step: {error}")
        else:
            dormant_csv = output_dir / "dormant_neuron_fraction_by_eval_step.csv"
            dormant_plot = output_dir / "dormant_neuron_fraction_by_eval_step.png"
            dormant.to_csv(dormant_csv, index=False)
            plot_dormant_neuron_fraction_by_eval_step(dormant, dormant_plot)
            output_paths.extend([dormant_csv, dormant_plot])

    print("\nSaved outputs:")
    for path in output_paths:
        print(f"- {path}")
    return output_paths


def analysis_output_dir(run_dirs: list[Path], output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if len(run_dirs) == 1:
        return run_dirs[0] / "analysis"
    return run_dirs[0].parent / "comparison_analysis"


def clear_previous_analysis_outputs(output_dir: Path) -> None:
    for pattern in ["*.png", "*.csv"]:
        for path in output_dir.glob(pattern):
            path.unlink()


def infer_mse_threshold(run_dirs: list[Path]) -> float:
    for run_dir in run_dirs:
        config = load_run_config(run_dir)
        analysis_config = config.get("analysis", {})
        threshold_config = analysis_config.get("steps_to_mse_threshold_by_task", {})
        if isinstance(threshold_config, dict):
            threshold = threshold_config.get("mse_threshold")
        else:
            threshold = None
        if threshold is None:
            # Backward compatibility for older run folders.
            threshold = analysis_config.get("mse_threshold")
        if threshold is not None:
            return float(threshold)
    return 0.01


def infer_plot_flags(run_dirs: list[Path]) -> dict[str, bool]:
    flags = {name: True for name in PLOT_NAMES}
    if PLOT_FLAGS is not None:
        flags.update({name: bool(PLOT_FLAGS.get(name, flags[name])) for name in PLOT_NAMES})
        return flags

    for run_dir in run_dirs:
        config = load_run_config(run_dir)
        if not config:
            continue
        analysis_config = config.get("analysis", {})
        for name in PLOT_NAMES:
            if name in analysis_config:
                value = analysis_config[name]
                if isinstance(value, dict):
                    flags[name] = bool(value.get("enabled", True))
                else:
                    flags[name] = bool(value)
        return flags
    return flags


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open() as handle:
        return yaml.safe_load(handle) or {}


def print_table(title: str, df: pd.DataFrame, max_rows: int = 12) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if df.empty:
        print("(empty)")
        return
    print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"... {len(df) - max_rows} more rows")


if __name__ == "__main__":
    main()
