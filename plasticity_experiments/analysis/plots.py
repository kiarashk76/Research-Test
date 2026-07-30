from __future__ import annotations

import os
from pathlib import Path
import tempfile

cache_root = Path(
    os.environ.setdefault(
        "XDG_CACHE_HOME",
        str(Path(tempfile.gettempdir()) / "plasticity-experiments-cache"),
    )
)
matplotlib_config_dir = Path(
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
)
matplotlib_config_dir.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd


def plot_latest_eval_mse_by_task(df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in grouped_curves(df):
        ordered = group.sort_values("task_index")
        (line,) = ax.plot(
            ordered["task_index"],
            ordered["eval_loss_mean"],
            label=label,
            linewidth=1.8,
        )
        fill_sem(
            ax,
            ordered["task_index"],
            ordered["eval_loss_mean"],
            ordered["eval_loss_sem"],
            line.get_color(),
        )

    ax.set_xlabel("task number")
    ax.set_ylabel("Latest Eval MSE")
    ax.set_title("Latest Eval MSE by task")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_eval_mse_by_eval_step(df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, group in grouped_curves(df):
        ordered = group.sort_values("eval_step")
        (line,) = ax.plot(
            ordered["eval_step"],
            ordered["eval_loss_mean"],
            label=label,
            linewidth=1.8,
        )
        fill_sem(
            ax,
            ordered["eval_step"],
            ordered["eval_loss_mean"],
            ordered["eval_loss_sem"],
            line.get_color(),
        )

    for boundary in task_boundaries(df):
        ax.axvline(boundary, color="0.35", alpha=0.25, linewidth=0.9)

    ax.set_xlabel("eval step")
    ax.set_ylabel("Eval MSE")
    ax.set_title("Eval MSE across eval steps")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_eval_mse_auc_by_task(df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in grouped_curves(df):
        ordered = group.sort_values("task_index")
        (line,) = ax.plot(
            ordered["task_index"],
            ordered["eval_mse_auc_mean"],
            label=label,
            linewidth=1.8,
        )
        fill_sem(
            ax,
            ordered["task_index"],
            ordered["eval_mse_auc_mean"],
            ordered["eval_mse_auc_sem"],
            line.get_color(),
        )

    ax.set_xlabel("task number")
    ax.set_ylabel("AUC for Eval MSE")
    ax.set_title("AUC for Eval MSE by task")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_steps_to_mse_threshold_by_task(
    df: pd.DataFrame,
    output_path: str | Path,
    mse_threshold: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in grouped_curves(df):
        ordered = group.sort_values("task_index")
        (line,) = ax.plot(
            ordered["task_index"],
            ordered["steps_to_threshold_mean"],
            label=label,
            linewidth=1.8,
        )
        fill_sem(
            ax,
            ordered["task_index"],
            ordered["steps_to_threshold_mean"],
            ordered["steps_to_threshold_sem"],
            line.get_color(),
        )

    ax.set_xlabel("task number")
    ax.set_ylabel("Training steps to threshold")
    ax.set_title(f"Steps to Eval MSE <= {mse_threshold:g} by task")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_activation_erank_by_eval_step(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    plot_layer_metric_by_eval_step(
        df=df,
        output_path=output_path,
        value_col="normalized_effective_rank_mean",
        sem_col="normalized_effective_rank_sem",
        y_label="Normalized effective rank",
        title="Normalized effective representation rank by eval step",
    )


def plot_dormant_neuron_fraction_by_eval_step(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    threshold = (
        float(df["activation_threshold"].dropna().iloc[0])
        if "activation_threshold" in df.columns
        and not df["activation_threshold"].dropna().empty
        else 0.01
    )
    plot_layer_metric_by_eval_step(
        df=df,
        output_path=output_path,
        value_col="dormant_fraction_mean",
        sem_col="dormant_fraction_sem",
        y_label="Dormant neuron fraction",
        title=f"Dormant neuron fraction by eval step (mean |activation| < {threshold:g})",
    )


def plot_layer_metric_by_eval_step(
    *,
    df: pd.DataFrame,
    output_path: str | Path,
    value_col: str,
    sem_col: str,
    y_label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, group in grouped_layer_curves(df):
        ordered = group.sort_values("eval_step")
        (line,) = ax.plot(
            ordered["eval_step"],
            ordered[value_col],
            label=label,
            linewidth=1.8,
        )
        fill_sem(
            ax,
            ordered["eval_step"],
            ordered[value_col],
            ordered[sem_col],
            line.get_color(),
        )

    for boundary in task_boundaries(df):
        ax.axvline(boundary, color="0.35", alpha=0.25, linewidth=0.9)

    ax.set_xlabel("eval step")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def grouped_curves(df: pd.DataFrame):
    multiple_runs = df["run_label"].nunique() > 1
    for keys, group in df.groupby(["run_label", "intervention_type"]):
        run_label, intervention_type = keys
        label = (
            f"{run_label} / {intervention_type}"
            if multiple_runs
            else str(intervention_type)
        )
        yield label, group


def grouped_layer_curves(df: pd.DataFrame):
    multiple_runs = df["run_label"].nunique() > 1
    for keys, group in df.groupby(["run_label", "intervention_type", "layer_index"]):
        run_label, intervention_type, layer_index = keys
        layer_label = f"layer {int(layer_index)}"
        label = (
            f"{run_label} / {intervention_type} / {layer_label}"
            if multiple_runs
            else f"{intervention_type} / {layer_label}"
        )
        yield label, group


def fill_sem(ax, x, mean, sem, color: str) -> None:
    sem = sem.fillna(0.0)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.12)


def task_boundaries(df: pd.DataFrame) -> list[int]:
    first_curve = next(iter(df.groupby(["run_label", "intervention_type"])))[1]
    boundaries = (
        first_curve.groupby("task_index")["eval_step"]
        .min()
        .sort_index()
    )
    return [
        int(boundary)
        for task_index, boundary in boundaries.items()
        if int(task_index) != 0
    ]
