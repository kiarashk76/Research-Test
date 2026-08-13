"""Plot diagnostic trajectories aligned to detected plasticity-loss onset.

This script only reads saved experiment metadata and method-result files. It
does not retrain models or modify saved experiments. The current result format
supports task-level alignment because saved global and per-layer arrays have
shape ``[num_runs, num_tasks]``.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
except ImportError:  # Keep the script usable without tqdm.
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .analyze_metric_predictors import (
        LOSS_METRICS,
        _as_task_matrix,
        _compatible,
        _metric_scopes,
        _metric_values,
        detect_loss_onset,
        discover_experiment_dirs,
        load_experiments,
    )
except ImportError:  # Supports direct execution from plotting/.
    from analyze_metric_predictors import (
        LOSS_METRICS,
        _as_task_matrix,
        _compatible,
        _metric_scopes,
        _metric_values,
        detect_loss_onset,
        discover_experiment_dirs,
        load_experiments,
    )


# ---------------------------------------------------------------------------
# Editable configuration
# ---------------------------------------------------------------------------

EXPERIMENT_ROOTS = ["outputs/nonlinear_xfixed"]
EXPERIMENT_DIRS = []
METHODS_TO_INCLUDE = [
    "backprop",
    "continual_backprop",
    "redo",
    "l2_init",
]
LOSS_REFERENCE_METHOD = "fresh"
METRICS_TO_INCLUDE = None  # None discovers every diagnostic metric.
SCOPES_TO_INCLUDE = None  # None includes global and every available layer.

PRE_ONSET_TASKS = 50
POST_ONSET_TASKS = 30
BASELINE_START = -50
BASELINE_END = -30
MIN_TRAJECTORIES = 3
NORMALIZATION = "baseline_relative"
# Supported: "baseline_delta", "baseline_relative", "zscore_pre_onset", "none".
AGGREGATION = "median"  # Supported: "median", "mean".
SHOW_INDIVIDUAL_TRAJECTORIES = False
POOL_METHODS = False
OUTPUT_DIR = "outputs/metrics_aligned_to_loss_onset"

TREND_START = -30
TREND_END = -1
EPSILON = 1e-12


def _safe_name(value):
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _discover_metrics_and_scopes(experiments, selected_methods):
    metric_names = set()
    scopes_by_metric = defaultdict(set)
    for experiment in experiments:
        for method_name in selected_methods:
            payload = experiment["methods"].get(method_name)
            if payload is None:
                continue
            results = payload.get("results", {})
            for metric_name in results:
                if metric_name in LOSS_METRICS:
                    continue
                metric_names.add(metric_name)
                scopes_by_metric[metric_name].update(_metric_scopes(results, metric_name))
    if METRICS_TO_INCLUDE is not None:
        metric_names.intersection_update(METRICS_TO_INCLUDE)
    if SCOPES_TO_INCLUDE is not None:
        selected_scopes = set(SCOPES_TO_INCLUDE)
        scopes_by_metric = {
            metric: scopes & selected_scopes
            for metric, scopes in scopes_by_metric.items()
        }
    return sorted(metric_names), {
        metric: sorted(scopes)
        for metric, scopes in scopes_by_metric.items()
        if scopes
    }


def _aligned_values(values, onset_task):
    relative_tasks = np.arange(-PRE_ONSET_TASKS, POST_ONSET_TASKS + 1)
    aligned = np.full(relative_tasks.shape, np.nan, dtype=np.float64)
    for index, relative_task in enumerate(relative_tasks):
        task = int(onset_task + relative_task)
        if 0 <= task < len(values):
            aligned[index] = values[task]
    return relative_tasks, aligned


def _normalize(aligned, relative_tasks):
    if NORMALIZATION == "none":
        return aligned.copy(), "none"
    baseline_mask = (relative_tasks >= BASELINE_START) & (relative_tasks <= BASELINE_END)
    baseline_values = aligned[baseline_mask & np.isfinite(aligned)]
    if baseline_values.size < 2:
        return None, None
    baseline = float(np.median(baseline_values))
    if NORMALIZATION == "baseline_delta":
        return aligned - baseline, "baseline_delta"
    if NORMALIZATION == "baseline_relative":
        denominator = abs(baseline) + EPSILON
        if denominator <= EPSILON * 10:
            return aligned - baseline, "baseline_delta_fallback"
        return (aligned - baseline) / denominator, "baseline_relative"
    if NORMALIZATION == "zscore_pre_onset":
        scale = float(np.std(baseline_values))
        if scale <= EPSILON:
            return aligned - baseline, "baseline_delta_fallback"
        return (aligned - baseline) / scale, "zscore_pre_onset"
    raise ValueError(
        "NORMALIZATION must be one of: baseline_delta, baseline_relative, "
        "zscore_pre_onset, none"
    )


def _loss_target(experiment, method_name):
    payload = experiment["methods"].get(method_name)
    if payload is None:
        return None, None
    method_loss = _as_task_matrix(_metric_values(payload.get("results", {}), "final_loss", "global"))
    if method_loss is None:
        return None, None
    fresh_payload = experiment["methods"].get(LOSS_REFERENCE_METHOD)
    if (
        fresh_payload is not None
        and method_name != LOSS_REFERENCE_METHOD
        and _compatible(
            method_loss,
            _metric_values(fresh_payload.get("results", {}), "final_loss", "global"),
        )
    ):
        fresh_loss = _as_task_matrix(
            _metric_values(fresh_payload.get("results", {}), "final_loss", "global")
        )
        return method_loss - fresh_loss, "fresh_gap"
    if method_name == LOSS_REFERENCE_METHOD and fresh_payload is not None:
        # Fresh compared with itself has no meaningful gap and must not be
        # treated as a plasticity-loss trajectory.
        return None, None
    return method_loss, "raw_loss"


def collect_aligned_trajectories(experiments, selected_methods, metric_names, scopes_by_metric):
    """Collect normalized aligned metric and loss-gap trajectories."""
    metric_rows = []
    loss_rows = []
    onset_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for experiment in tqdm(experiments, desc="Collecting aligned trajectories", unit="experiment"):
        for method_name in tqdm(
            selected_methods,
            desc=f"Methods ({experiment['architecture']})",
            unit="method",
            leave=False,
        ):
            if method_name not in experiment["methods"]:
                continue
            loss_target, loss_reference_mode = _loss_target(experiment, method_name)
            if loss_target is None:
                continue
            method_payload = experiment["methods"][method_name]
            method_results = method_payload.get("results", {})
            method_label = method_payload.get("method_config", {}).get("label", method_name)
            for run_index in range(loss_target.shape[0]):
                total_counts[method_name] += 1
                onset = detect_loss_onset(loss_target[run_index])
                if onset["task"] is None:
                    continue
                onset_counts[method_name] += 1
                relative_tasks, aligned_loss = _aligned_values(loss_target[run_index], onset["task"])
                loss_rows.extend(
                    {
                        "dataset": experiment["config"].get("dataset", "unknown"),
                        "experiment": experiment["experiment"],
                        "experiment_id": experiment["experiment_id"],
                        "architecture": experiment["architecture"],
                        "architecture_key": experiment["architecture_key"],
                        "method": method_name,
                        "run": run_index,
                        "loss_reference_mode": loss_reference_mode,
                        "loss_onset_task": onset["task"],
                        "relative_task": int(relative_task),
                        "loss_gap": value,
                    }
                    for relative_task, value in zip(relative_tasks, aligned_loss)
                    if np.isfinite(value)
                )

                metric_items = [
                    (metric_name, scope)
                    for metric_name in metric_names
                    for scope in scopes_by_metric.get(metric_name, [])
                ]
                for metric_name, scope in metric_items:
                    values = _as_task_matrix(_metric_values(method_results, metric_name, scope))
                    if values is None or run_index >= values.shape[0] or values.shape[1] != loss_target.shape[1]:
                        continue
                    relative_tasks, aligned = _aligned_values(values[run_index], onset["task"])
                    normalized, normalization_used = _normalize(aligned, relative_tasks)
                    if normalized is None:
                        continue
                    metric_label = method_payload.get("metrics", {}).get(metric_name, {}).get("label", metric_name)
                    for relative_task, value in zip(relative_tasks, normalized):
                        if not np.isfinite(value):
                            continue
                        metric_rows.append(
                            {
                                "dataset": experiment["config"].get("dataset", "unknown"),
                                "experiment": experiment["experiment"],
                                "experiment_id": experiment["experiment_id"],
                                "experiment_dir": experiment["experiment_dir"],
                                "architecture": experiment["architecture"],
                                "architecture_key": experiment["architecture_key"],
                                "method": method_name,
                                "method_label": method_label,
                                "run": run_index,
                                "metric": metric_name,
                                "metric_label": metric_label,
                                "scope": scope,
                                "loss_onset_task": onset["task"],
                                "loss_reference_mode": loss_reference_mode,
                                "relative_task": int(relative_task),
                                "normalized_metric_value": value,
                                "normalization_used": normalization_used,
                            }
                        )
    return metric_rows, loss_rows, dict(onset_counts), dict(total_counts)


def _aggregate(values):
    values = np.asarray(values, dtype=np.float64)
    if AGGREGATION == "median":
        return float(np.median(values))
    if AGGREGATION == "mean":
        return float(np.mean(values))
    raise ValueError("AGGREGATION must be 'median' or 'mean'")


def _summary_rows(metric_rows):
    curve_key = "method" if not POOL_METHODS else None
    groups = defaultdict(list)
    for row in metric_rows:
        key = (row["metric"], row["scope"], row[curve_key] if curve_key else "pooled")
        groups[key].append(row)
    summaries = []
    for (metric, scope, curve_method), group in groups.items():
        trajectory_ids = {
            (row["experiment_id"], row["method"], row["run"])
            for row in group
        }
        experiments = {row["experiment_id"] for row in group}
        architectures = {row["architecture_key"] for row in group}
        for relative_task in range(-PRE_ONSET_TASKS, POST_ONSET_TASKS + 1):
            values = [
                float(row["normalized_metric_value"])
                for row in group
                if row["relative_task"] == relative_task
            ]
            if not values:
                continue
            summaries.append(
                {
                    "dataset": group[0]["dataset"],
                    "method": curve_method,
                    "metric": metric,
                    "scope": scope,
                    "relative_task": relative_task,
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                    "aggregation": AGGREGATION,
                    "num_values": len(values),
                    "num_trajectories": len(trajectory_ids),
                    "num_runs": len({(row["experiment_id"], row["method"], row["run"]) for row in group}),
                    "num_experiments": len(experiments),
                    "num_architectures": len(architectures),
                }
            )
    return summaries


def _trend_summary(metric_rows):
    trajectories = defaultdict(list)
    for row in metric_rows:
        key = (
            row["dataset"], row["experiment_id"], row["architecture_key"],
            row["method"], row["metric"], row["scope"], row["run"],
        )
        trajectories[key].append((row["relative_task"], row["normalized_metric_value"]))
    slopes = defaultdict(list)
    for key, values in trajectories.items():
        selected = [
            (task, value) for task, value in values
            if TREND_START <= task <= TREND_END and np.isfinite(value)
        ]
        if len(selected) < 3:
            continue
        tasks, metric_values = zip(*selected)
        slope = float(np.polyfit(np.asarray(tasks), np.asarray(metric_values), 1)[0])
        group_key = key[:6]
        slopes[group_key].append(slope)

    rows = []
    for (dataset, experiment_id, architecture_key, method, metric, scope), values in slopes.items():
        values = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "experiment_id": experiment_id,
                "architecture_key": architecture_key,
                "method": method,
                "metric": metric,
                "scope": scope,
                "median_pre_onset_slope": float(np.median(values)),
                "q25_slope": float(np.percentile(values, 25)),
                "q75_slope": float(np.percentile(values, 75)),
                "fraction_positive_slope": float(np.mean(values > 0)),
                "fraction_negative_slope": float(np.mean(values < 0)),
                "num_trajectories": len(values),
            }
        )
    return rows


def _plot_metric_scope(metric, scope, metric_rows, summary_rows, output_dir):
    curve_method = "method" if not POOL_METHODS else None
    matching = [
        row for row in metric_rows
        if row["metric"] == metric and row["scope"] == scope
    ]
    if not matching:
        return False
    groups = defaultdict(list)
    for row in matching:
        groups[row[curve_method] if curve_method else "pooled"].append(row)
    groups = {
        method: rows
        for method, rows in groups.items()
        if len({(row["experiment_id"], row["method"], row["run"]) for row in rows}) >= MIN_TRAJECTORIES
    }
    if not groups:
        return False

    fig, axis = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(groups))))
    for color, (method, rows) in zip(colors, sorted(groups.items())):
        trajectory_ids = {(row["experiment_id"], row["method"], row["run"]) for row in rows}
        by_task = defaultdict(list)
        for row in rows:
            by_task[row["relative_task"]].append(row["normalized_metric_value"])
        x = np.arange(-PRE_ONSET_TASKS, POST_ONSET_TASKS + 1)
        median = np.full(x.shape, np.nan)
        q25 = np.full(x.shape, np.nan)
        q75 = np.full(x.shape, np.nan)
        for index, relative_task in enumerate(x):
            values = by_task.get(int(relative_task), [])
            if values:
                median[index] = _aggregate(values)
                q25[index] = np.percentile(values, 25)
                q75[index] = np.percentile(values, 75)
        if SHOW_INDIVIDUAL_TRAJECTORIES:
            for trajectory_id in sorted(trajectory_ids):
                trajectory = [
                    row for row in rows
                    if (row["experiment_id"], row["method"], row["run"]) == trajectory_id
                ]
                values_by_task = {row["relative_task"]: row["normalized_metric_value"] for row in trajectory}
                axis.plot(
                    list(values_by_task),
                    list(values_by_task.values()),
                    color=color,
                    alpha=0.10,
                    linewidth=0.7,
                )
        label = f"{method} (n={len(trajectory_ids)})"
        axis.plot(x, median, color=color, linewidth=2, label=label)
        axis.fill_between(x, q25, q75, color=color, alpha=0.20)

    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
    axis.set_xlabel("Tasks relative to detected loss onset")
    axis.set_ylabel(f"Metric ({NORMALIZATION})")
    axis.set_title(f"{metric} / {scope} aligned to plasticity-loss onset")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    total_trajectories = len({(row["experiment_id"], row["method"], row["run"]) for row in matching})
    subtitle = (
        f"methods: {', '.join(sorted(groups))} | n={total_trajectories} trajectories | "
        f"experiments={len({row['experiment_id'] for row in matching})} | "
        f"architectures={len({row['architecture_key'] for row in matching})}"
    )
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    filename = f"{_safe_name(metric)}__{_safe_name(scope)}.png"
    fig.savefig(os.path.join(output_dir, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_loss_gap(loss_rows, output_dir):
    groups = defaultdict(list)
    for row in loss_rows:
        groups[row["method"]].append(row)
    groups = {
        method: rows
        for method, rows in groups.items()
        if len({(row["experiment_id"], row["method"], row["run"]) for row in rows}) >= MIN_TRAJECTORIES
    }
    if not groups:
        return False
    fig, axis = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(groups))))
    x = np.arange(-PRE_ONSET_TASKS, POST_ONSET_TASKS + 1)
    for color, (method, rows) in zip(colors, sorted(groups.items())):
        by_task = defaultdict(list)
        for row in rows:
            by_task[row["relative_task"]].append(row["loss_gap"])
        median = np.full(x.shape, np.nan)
        q25 = np.full(x.shape, np.nan)
        q75 = np.full(x.shape, np.nan)
        for index, relative_task in enumerate(x):
            values = by_task.get(int(relative_task), [])
            if values:
                median[index] = np.median(values)
                q25[index] = np.percentile(values, 25)
                q75[index] = np.percentile(values, 75)
        trajectory_count = len({(row["experiment_id"], row["method"], row["run"]) for row in rows})
        axis.plot(x, median, color=color, linewidth=2, label=f"{method} (n={trajectory_count})")
        axis.fill_between(x, q25, q75, color=color, alpha=0.20)
    axis.axvline(0, color="black", linestyle="--")
    axis.axhline(0, color="black", linestyle=":", alpha=0.7)
    axis.set_xlabel("Tasks relative to detected loss onset")
    axis.set_ylabel("Loss gap: method loss - Fresh loss")
    axis.set_title("Loss gap aligned to detected plasticity-loss onset")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "aligned_loss_gap.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def analyze_dataset(experiments, dataset, output_dir, selected_methods):
    metric_names, scopes_by_metric = _discover_metrics_and_scopes(experiments, selected_methods)
    metric_rows, loss_rows, onset_counts, total_counts = collect_aligned_trajectories(
        experiments,
        selected_methods,
        metric_names,
        scopes_by_metric,
    )
    os.makedirs(output_dir, exist_ok=True)
    summaries = _summary_rows(metric_rows)
    trends = _trend_summary(metric_rows)
    _write_csv(os.path.join(output_dir, "aligned_metric_trajectories.csv"), metric_rows)
    _write_csv(os.path.join(output_dir, "aligned_metric_summary.csv"), summaries)
    _write_csv(os.path.join(output_dir, "pre_onset_trend_summary.csv"), trends)
    _write_csv(os.path.join(output_dir, "aligned_loss_trajectories.csv"), loss_rows)

    plotted = 0
    for metric_name in metric_names:
        for scope in scopes_by_metric.get(metric_name, []):
            if _plot_metric_scope(metric_name, scope, metric_rows, summaries, output_dir):
                plotted += 1
    _plot_loss_gap(loss_rows, output_dir)

    with open(os.path.join(output_dir, "analysis_metadata.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"dataset={dataset}\n")
        handle.write(f"normalization={NORMALIZATION}\n")
        handle.write(f"aggregation={AGGREGATION}\n")
        handle.write(f"pool_methods={POOL_METHODS}\n")
        handle.write(f"baseline_start={BASELINE_START}\n")
        handle.write(f"baseline_end={BASELINE_END}\n")
        handle.write(f"experiments={len(experiments)}\n")
        handle.write(f"methods={','.join(selected_methods)}\n")

    print(f"Analyzed {dataset}")
    for method_name in selected_methods:
        print(f"{method_name}: {onset_counts.get(method_name, 0)} / {total_counts.get(method_name, 0)} runs with detected loss onset")
    for metric_name, scope in sorted({(row["metric"], row["scope"]) for row in metric_rows}):
        matching = [row for row in metric_rows if row["metric"] == metric_name and row["scope"] == scope]
        trajectory_count = len({(row["experiment_id"], row["method"], row["run"]) for row in matching})
        trend_rows = [row for row in trends if row["metric"] == metric_name and row["scope"] == scope]
        slope_values = [row["median_pre_onset_slope"] for row in trend_rows]
        print(
            f"{metric_name} / {scope}: {trajectory_count} trajectories, "
            f"median pre-onset slope={slope_values[0] if slope_values else 'n/a'}"
        )
    print(f"Wrote {plotted} metric/scope plots to {output_dir}")
    return {
        "metric_rows": metric_rows,
        "loss_rows": loss_rows,
        "summary_rows": summaries,
        "trend_rows": trends,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", action="append", dest="experiment_dirs")
    parser.add_argument("--root", action="append", dest="roots")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    experiment_dirs = discover_experiment_dirs(args.experiment_dirs or EXPERIMENT_DIRS, args.roots or EXPERIMENT_ROOTS)
    if not experiment_dirs:
        raise SystemExit(
            "No experiments found. Set EXPERIMENT_ROOTS/EXPERIMENT_DIRS or pass "
            "--root/--experiment-dir."
        )
    selected_methods = args.methods or METHODS_TO_INCLUDE
    methods_to_load = list(dict.fromkeys(selected_methods + [LOSS_REFERENCE_METHOD]))
    loaded = load_experiments(experiment_dirs, methods_to_load)
    if not loaded:
        raise SystemExit("No compatible experiment/method result files were found.")

    by_dataset = defaultdict(list)
    for experiment in loaded:
        dataset = experiment["config"].get("dataset", "unknown")
        by_dataset[dataset].append(experiment)
    if len(by_dataset) > 1:
        print(f"Detected multiple datasets; writing separate output directories: {sorted(by_dataset)}")

    results = {}
    for dataset, experiments in by_dataset.items():
        dataset_output = os.path.join(args.output_dir, _safe_name(dataset))
        results[dataset] = analyze_dataset(experiments, dataset, dataset_output, selected_methods)
    return results


if __name__ == "__main__":
    main()
