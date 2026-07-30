from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_run_csvs(run_dir: str | Path) -> pd.DataFrame:
    paths = sorted(Path(run_dir).glob("seed_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No seed CSVs found in {run_dir}")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def prepare_supervised_eval_df(run_dir: str | Path) -> pd.DataFrame:
    df = load_run_csvs(run_dir)
    if "eval_loss" not in df.columns:
        raise ValueError(f"{run_dir} is not a supervised run: expected eval_loss column.")

    required = {"seed", "task_index", "update_within_task", "global_update", "eval_loss"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{run_dir} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["run_label"] = Path(run_dir).name
    if "intervention_type" not in df.columns:
        df["intervention_type"] = df["model"] if "model" in df.columns else "unknown"

    sort_cols = ["run_label", "intervention_type", "seed", "global_update"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["eval_step"] = (
        df.groupby(["run_label", "intervention_type", "seed"]).cumcount() + 1
    )
    return df


def latest_eval_mse_by_task(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["run_label", "intervention_type", "seed", "task_index"]
    latest = (
        df.sort_values("update_within_task")
        .groupby(group_cols, as_index=False)
        .tail(1)
    )
    return (
        latest.groupby(["run_label", "intervention_type", "task_index"])
        .agg(
            eval_loss_mean=("eval_loss", "mean"),
            eval_loss_sem=("eval_loss", "sem"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["run_label", "intervention_type", "task_index"])
    )


def eval_mse_by_eval_step(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["run_label", "intervention_type", "eval_step"])
        .agg(
            task_index=("task_index", "first"),
            global_update_mean=("global_update", "mean"),
            update_within_task=("update_within_task", "first"),
            eval_loss_mean=("eval_loss", "mean"),
            eval_loss_sem=("eval_loss", "sem"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["run_label", "intervention_type", "eval_step"])
    )


def eval_mse_auc_by_task(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["run_label", "intervention_type", "seed", "task_index"]
    for keys, group in df.groupby(group_cols):
        ordered = group.sort_values("update_within_task")
        auc = trapezoid_area(
            ordered["update_within_task"].tolist(),
            ordered["eval_loss"].tolist(),
        )
        run_label, intervention_type, seed, task_index = keys
        rows.append(
            {
                "run_label": run_label,
                "intervention_type": intervention_type,
                "seed": seed,
                "task_index": task_index,
                "eval_mse_auc": auc,
            }
        )
    per_seed = pd.DataFrame(rows)
    return (
        per_seed.groupby(["run_label", "intervention_type", "task_index"])
        .agg(
            eval_mse_auc_mean=("eval_mse_auc", "mean"),
            eval_mse_auc_sem=("eval_mse_auc", "sem"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["run_label", "intervention_type", "task_index"])
    )


def steps_to_mse_threshold_by_task(
    df: pd.DataFrame,
    mse_threshold: float,
) -> pd.DataFrame:
    rows = []
    group_cols = ["run_label", "intervention_type", "seed", "task_index"]
    for keys, group in df.groupby(group_cols):
        ordered = group.sort_values("update_within_task")
        reached = ordered[ordered["eval_loss"] <= mse_threshold]
        steps_to_threshold = (
            float(reached.iloc[0]["update_within_task"])
            if not reached.empty
            else float("nan")
        )
        run_label, intervention_type, seed, task_index = keys
        rows.append(
            {
                "run_label": run_label,
                "intervention_type": intervention_type,
                "seed": seed,
                "task_index": task_index,
                "mse_threshold": mse_threshold,
                "steps_to_threshold": steps_to_threshold,
                "reached_threshold": not reached.empty,
            }
        )
    per_seed = pd.DataFrame(rows)
    return (
        per_seed.groupby(["run_label", "intervention_type", "task_index"])
        .agg(
            mse_threshold=("mse_threshold", "first"),
            steps_to_threshold_mean=("steps_to_threshold", "mean"),
            steps_to_threshold_sem=("steps_to_threshold", "sem"),
            n_seeds=("seed", "nunique"),
            n_reached=("reached_threshold", "sum"),
        )
        .reset_index()
        .sort_values(["run_label", "intervention_type", "task_index"])
    )


def trapezoid_area(x_values: list[float], y_values: list[float]) -> float:
    if not x_values:
        return float("nan")
    if len(x_values) == 1:
        return float(y_values[0])

    area = 0.0
    for i in range(1, len(x_values)):
        width = float(x_values[i]) - float(x_values[i - 1])
        height = (float(y_values[i]) + float(y_values[i - 1])) / 2.0
        area += width * height
    return area
