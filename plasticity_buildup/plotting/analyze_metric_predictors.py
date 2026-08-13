"""Analyze diagnostic metrics as early-warning signals for plasticity loss.

This module only reads saved experiment metadata and method-result pickle files.
It does not retrain models, regenerate tasks, or modify existing result files.

The saved result format currently contains one value per task:

* global arrays: ``[num_runs, num_tasks]``;
* per-layer arrays: ``[num_runs, num_tasks]``;
* ``method_statistics``: per-task intervention statistics, not finer-grained
  diagnostic trajectories.

Run this file directly after editing the configuration constants below, or
import its functions from a notebook.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # Keep the analyzer usable if tqdm is unavailable.
    def tqdm(iterable, **kwargs):
        return iterable

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..experiments.workflows import load_shared_experiment
    from ..utils.io import load_method_results
except ImportError:  # Supports `python plotting/analyze_metric_predictors.py`.
    from experiments.workflows import load_shared_experiment
    from utils.io import load_method_results


# ---------------------------------------------------------------------------
# Editable analysis configuration
# ---------------------------------------------------------------------------

EXPERIMENT_DIRS = []
EXPERIMENT_ROOTS = ["outputs"]
METHODS_TO_ANALYZE = [
    "backprop",
    "fresh",
    "continual_backprop",
    "redo",
    "l2_init",
    "l2",
    "low_gradient_reset",
    "shrink_and_perturb",
]
BASELINE_METHOD = "fresh"
OUTPUT_DIR = "outputs/metric_predictor_analysis"

USE_FRESH_NORMALIZATION = True
BASELINE_TASKS = 20
SMOOTHING_WINDOW = 5
ONSET_CONSECUTIVE_TASKS = 5
LOSS_ABSOLUTE_THRESHOLD = 0.05
LOSS_RELATIVE_THRESHOLD = 0.25
METRIC_ABSOLUTE_THRESHOLD = 1e-4
METRIC_RELATIVE_THRESHOLD = 0.20
LAGS = (0, 1, 2, 5, 10, 20)
MIN_CORRELATION_POINTS = 5
CORRELATION_ONLY_ONSET_RUNS = True

ALIGNMENT_WINDOW = 30
TOP_N_PLOTS = 3
PLOT_HEATMAP_METHOD = "backprop"


LOSS_METRICS = {"first_loss", "average_loss", "final_loss"}
PREFERRED_METHODS = ("fresh", "continual_backprop", "redo", "backprop")


def _architecture_info(config):
    """Return human-readable and stable architecture identifiers."""
    architecture_config = {
        "input_dim": config.get("input_dim"),
        "hidden_dims": config.get("hidden_dims"),
        "output_dim": config.get("output_dim"),
    }
    architecture_key = json.dumps(architecture_config, sort_keys=True, separators=(",", ":"))
    architecture_label = f"hidden={architecture_config['hidden_dims']}"
    return architecture_label, architecture_key, architecture_config


def _experiment_label(experiment_dir, metadata):
    config = metadata.get("config", {})
    dataset = config.get("dataset")
    architecture, _, _ = _architecture_info(config)
    name = os.path.basename(os.path.normpath(experiment_dir)) or experiment_dir
    return f"{name} | {dataset} | {architecture}"


def discover_experiment_dirs(experiment_dirs=None, roots=None):
    """Discover directories containing metadata.json and method_results."""
    explicit = experiment_dirs if experiment_dirs is not None else EXPERIMENT_DIRS
    if explicit:
        return sorted(dict.fromkeys(os.path.abspath(path) for path in explicit))

    roots = roots if roots is not None else EXPERIMENT_ROOTS
    discovered = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for directory, _, files in os.walk(root):
            if "metadata.json" in files and os.path.isdir(os.path.join(directory, "method_results")):
                discovered.append(os.path.abspath(directory))
    return sorted(dict.fromkeys(discovered))


def _available_method_names(experiment_dir):
    method_dir = os.path.join(experiment_dir, "method_results")
    if not os.path.isdir(method_dir):
        return []
    return sorted(
        filename[:-4]
        for filename in os.listdir(method_dir)
        if filename.endswith(".pkl")
    )


def load_experiments(experiment_dirs, method_names=None):
    """Load compatible saved experiment metadata and requested method files."""
    loaded = []
    requested = list(method_names) if method_names else None
    for experiment_dir in tqdm(experiment_dirs, desc="Loading experiments", unit="experiment"):
        metadata = load_shared_experiment(experiment_dir)
        available = _available_method_names(experiment_dir)
        names = requested or available
        payloads = []
        skipped = []
        experiment_name = os.path.basename(os.path.normpath(experiment_dir))
        for method_name in tqdm(
            names,
            desc=f"Loading methods ({experiment_name})",
            unit="method",
            leave=False,
        ):
            if method_name not in available:
                skipped.append(method_name)
                continue
            try:
                payload = load_method_results(
                    os.path.join(experiment_dir, "method_results"),
                    [method_name],
                    expected_experiment_id=metadata["experiment_id"],
                )[0]
            except (FileNotFoundError, ValueError, KeyError) as error:
                skipped.append(f"{method_name}: {error}")
                continue
            payloads.append(payload)

        if not payloads:
            continue
        config = metadata.get("config", {})
        architecture, architecture_key, architecture_config = _architecture_info(config)
        loaded.append(
            {
                "experiment_dir": experiment_dir,
                "experiment_id": metadata.get("experiment_id", experiment_dir),
                "experiment": _experiment_label(experiment_dir, metadata),
                "metadata": metadata,
                "config": config,
                "architecture": architecture,
                "architecture_key": architecture_key,
                "architecture_config": architecture_config,
                "methods": {payload["method_name"]: payload for payload in payloads},
                "skipped_methods": skipped,
            }
        )
    return loaded


def _as_task_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        return None
    if not np.isfinite(values).all():
        return None
    return values


def _metric_values(results, metric_name, scope):
    metric = results.get(metric_name)
    if metric is None:
        return None
    if scope == "global":
        return metric.get("global")
    return (metric.get("per_layer") or {}).get(scope)


def _metric_scopes(results, metric_name):
    metric = results.get(metric_name, {})
    scopes = []
    if _as_task_matrix(metric.get("global")) is not None:
        scopes.append("global")
    scopes.extend(sorted((metric.get("per_layer") or {}).keys()))
    return scopes


def _compatible(a, b):
    a = _as_task_matrix(a)
    b = _as_task_matrix(b)
    return a is not None and b is not None and a.shape == b.shape


def _moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size < 2:
        return values.copy()
    window = min(int(window), values.size)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _robust_scale(values):
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return max(abs(median), 1.4826 * mad, 1e-12)


def _threshold(values, absolute_threshold, relative_threshold):
    baseline_scale = _robust_scale(values)
    return max(float(absolute_threshold), float(relative_threshold) * baseline_scale)


def _first_consecutive(mask, start, count):
    count = max(1, int(count))
    for index in range(max(0, start), len(mask) - count + 1):
        if np.all(mask[index:index + count]):
            return index
    return None


def detect_loss_onset(values):
    """Detect sustained upward loss/gap departure from an early baseline."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        return {"task": None, "baseline": None, "threshold": None, "direction": None}
    baseline_count = min(BASELINE_TASKS, len(values))
    if len(values) <= baseline_count:
        return {"task": None, "baseline": None, "threshold": None, "direction": None}
    smoothed = _moving_average(values, SMOOTHING_WINDOW)
    baseline = float(np.median(smoothed[:baseline_count]))
    threshold = _threshold(
        smoothed[:baseline_count],
        LOSS_ABSOLUTE_THRESHOLD,
        LOSS_RELATIVE_THRESHOLD,
    )
    mask = smoothed >= baseline + threshold
    onset = _first_consecutive(mask, baseline_count, ONSET_CONSECUTIVE_TASKS)
    return {
        "task": onset,
        "baseline": baseline,
        "threshold": threshold,
        "direction": "increase" if onset is not None else None,
    }


def detect_metric_onset(values):
    """Detect the earliest sustained increase or decrease from baseline."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        return {"task": None, "baseline": None, "threshold": None, "direction": None}
    baseline_count = min(BASELINE_TASKS, len(values))
    if len(values) <= baseline_count:
        return {"task": None, "baseline": None, "threshold": None, "direction": None}
    smoothed = _moving_average(values, SMOOTHING_WINDOW)
    baseline_values = smoothed[:baseline_count]
    baseline = float(np.median(baseline_values))
    threshold = _threshold(
        baseline_values,
        METRIC_ABSOLUTE_THRESHOLD,
        METRIC_RELATIVE_THRESHOLD,
    )
    increase = _first_consecutive(
        smoothed >= baseline + threshold,
        baseline_count,
        ONSET_CONSECUTIVE_TASKS,
    )
    decrease = _first_consecutive(
        smoothed <= baseline - threshold,
        baseline_count,
        ONSET_CONSECUTIVE_TASKS,
    )
    candidates = [(task, "increase") for task in (increase,) if task is not None]
    candidates.extend((task, "decrease") for task in (decrease,) if task is not None)
    if not candidates:
        onset, direction = None, None
    else:
        onset, direction = min(candidates, key=lambda item: item[0])
    return {
        "task": onset,
        "baseline": baseline,
        "threshold": threshold,
        "direction": direction,
    }


def _baseline_delta(values):
    values = np.asarray(values, dtype=np.float64)
    count = min(BASELINE_TASKS, len(values))
    return values - np.median(values[:count])


def _rankdata(values):
    """Average-rank values without requiring scipy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def _correlation(x, y, kind):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < MIN_CORRELATION_POINTS:
        return None, len(x)
    if kind == "spearman":
        x, y = _rankdata(x), _rankdata(y)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None, len(x)
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def _lagged_pairs(predictor, target, lag):
    start = max(BASELINE_TASKS, int(lag))
    if start >= len(target) or start - lag >= len(predictor):
        return np.array([]), np.array([])
    x = predictor[start - lag:len(target) - lag]
    y = target[start:]
    count = min(len(x), len(y))
    return x[:count], y[:count]


def _transform(values, transform):
    if transform == "raw":
        return np.asarray(values, dtype=np.float64)
    if transform == "baseline_delta":
        return _baseline_delta(values)
    raise ValueError(f"Unknown transform: {transform}")


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


def _group(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in fields)].append(row)
    return groups


def _finite_values(rows, field):
    return np.asarray(
        [float(row[field]) for row in rows if row.get(field) not in (None, "")],
        dtype=np.float64,
    )


def _direction_consistency(rows, field):
    directions = [row.get(field) for row in rows if row.get(field)]
    if not directions:
        return None
    return max(directions.count("increase"), directions.count("decrease")) / len(directions)


def _aggregate_leads(rows, group_fields):
    summaries = []
    for key, group in _group(rows, group_fields).items():
        valid = [row for row in group if row.get("lead_tasks") not in (None, "")]
        leads = _finite_values(valid, "lead_tasks")
        architecture_groups = _group(valid, ["architecture_key"])
        architecture_medians = [
            float(np.median(_finite_values(architecture_rows, "lead_tasks")))
            for architecture_rows in architecture_groups.values()
            if _finite_values(architecture_rows, "lead_tasks").size
        ]
        positive_architectures = sum(value > 0 for value in architecture_medians)
        summary = {field: value for field, value in zip(group_fields, key)}
        first = group[0]
        summary.update(
            {
                "method_label": first.get("method_label", ""),
                "metric_label": first.get("metric_label", ""),
                "mean_lead_tasks": float(np.mean(leads)) if leads.size else None,
                "median_lead_tasks": float(np.median(leads)) if leads.size else None,
                "std_lead_tasks": float(np.std(leads)) if leads.size else None,
                "valid_runs": int(leads.size),
                "positive_lead_runs": int(np.sum(leads > 0)) if leads.size else 0,
                "positive_lead_fraction": float(np.mean(leads > 0)) if leads.size else None,
                "direction_consistency": _direction_consistency(valid, "metric_direction"),
                "loss_onset_runs": sum(row.get("loss_onset_task") not in (None, "") for row in group),
                "metric_onset_runs": sum(row.get("metric_onset_task") not in (None, "") for row in group),
                "total_runs": len(group),
                "architectures_with_valid_runs": len(architecture_medians),
                "architectures_with_positive_median_lead": positive_architectures,
                "architecture_consistency": (
                    positive_architectures / len(architecture_medians)
                    if architecture_medians else None
                ),
            }
        )
        summaries.append(summary)
    return summaries


def _aggregate_lags(run_rows):
    fields = [
        "method", "metric", "scope", "loss_mode", "diagnostic_mode",
        "predictor_transform", "target_transform", "lag", "correlation",
    ]
    summaries = []
    for key, group in _group(run_rows, fields).items():
        correlations = _finite_values(group, "correlation_value")
        architecture_groups = _group(
            [row for row in group if row.get("correlation_value") not in (None, "")],
            ["architecture_key"],
        )
        architecture_medians = [
            float(np.median(_finite_values(part, "correlation_value")))
            for part in architecture_groups.values()
            if _finite_values(part, "correlation_value").size
        ]
        positive_architectures = sum(value >= 0 for value in architecture_medians)
        summary = {field: value for field, value in zip(fields, key)}
        first = group[0]
        summary.update(
            {
                "method_label": first.get("method_label", ""),
                "metric_label": first.get("metric_label", ""),
                "mean_correlation": float(np.mean(correlations)) if correlations.size else None,
                "median_correlation": float(np.median(correlations)) if correlations.size else None,
                "std_correlation": float(np.std(correlations)) if correlations.size else None,
                "valid_runs": int(correlations.size),
                "positive_correlation_fraction": (
                    float(np.mean(correlations >= 0)) if correlations.size else None
                ),
                "direction_consistency": (
                    max(np.mean(correlations >= 0), np.mean(correlations < 0))
                    if correlations.size else None
                ),
                "architectures_with_valid_runs": len(architecture_medians),
                "architecture_consistency": (
                    max(positive_architectures, len(architecture_medians) - positive_architectures)
                    / len(architecture_medians)
                    if architecture_medians else None
                ),
            }
        )
        summaries.append(summary)
    return summaries


def _method_label(payload, method_name):
    return payload.get("method_config", {}).get("label", method_name)


def analyze_loaded_experiments(loaded):
    """Run onset and within-run lag analysis over already-loaded experiments."""
    onset_rows = []
    lag_run_rows = []
    trajectory_store = {}
    loss_store = {}
    available_diagnostic_metrics = set()
    fine_grained_statistics_found = False

    for experiment in tqdm(loaded, desc="Analyzing experiments", unit="experiment"):
        methods = experiment["methods"]
        fresh_payload = methods.get(BASELINE_METHOD)
        fresh_results = fresh_payload.get("results") if fresh_payload else None
        if fresh_payload and fresh_payload.get("method_statistics"):
            fine_grained_statistics_found = True

        metric_names = sorted({
            metric_name
            for payload in methods.values()
            for metric_name in payload.get("results", {})
            if metric_name not in LOSS_METRICS
        })
        available_diagnostic_metrics.update(metric_names)

        method_items = tqdm(
            methods.items(),
            desc=f"Analyzing methods ({experiment['architecture']})",
            unit="method",
            leave=False,
        )
        for method_name, payload in method_items:
            results = payload.get("results", {})
            method_label = _method_label(payload, method_name)
            method_loss = _as_task_matrix(_metric_values(results, "final_loss", "global"))
            if method_loss is None:
                continue
            loss_targets = [("raw_loss", method_loss)]
            if (
                USE_FRESH_NORMALIZATION
                and fresh_results is not None
                and method_name != BASELINE_METHOD
                and _compatible(method_loss, _metric_values(fresh_results, "final_loss", "global"))
            ):
                fresh_loss = _as_task_matrix(_metric_values(fresh_results, "final_loss", "global"))
                loss_targets.append(("fresh_gap", method_loss - fresh_loss))

            metric_scopes = sorted({
                (metric_name, scope)
                for metric_name in metric_names
                for scope in _metric_scopes(results, metric_name)
            })
            for loss_mode, loss_target_matrix in loss_targets:
                run_count = loss_target_matrix.shape[0]
                loss_onsets = {}
                for run_index in range(run_count):
                    loss_series = loss_target_matrix[run_index]
                    loss_onset = detect_loss_onset(loss_series)
                    loss_onsets[run_index] = loss_onset
                    loss_store[(experiment["experiment_id"], method_name, loss_mode, run_index)] = loss_series

                metric_items = tqdm(
                    metric_scopes,
                    desc=f"Metrics ({method_name}, {loss_mode})",
                    unit="metric",
                    leave=False,
                )
                for metric_name, scope in metric_items:
                    metric_values = _as_task_matrix(_metric_values(results, metric_name, scope))
                    if metric_values is None or metric_values.shape != loss_target_matrix.shape:
                        continue
                    metric_label = (
                        payload.get("metrics", {}).get(metric_name, {}).get("label", metric_name)
                    )
                    diagnostic_modes = [("raw", metric_values)]
                    if (
                        USE_FRESH_NORMALIZATION
                        and fresh_results is not None
                        and method_name != BASELINE_METHOD
                        and _compatible(metric_values, _metric_values(fresh_results, metric_name, scope))
                    ):
                        fresh_metric = _as_task_matrix(_metric_values(fresh_results, metric_name, scope))
                        diagnostic_modes.append(("fresh_delta", metric_values - fresh_metric))

                    for diagnostic_mode, diagnostic_matrix in diagnostic_modes:
                        for run_index in range(min(run_count, diagnostic_matrix.shape[0])):
                            metric_series = diagnostic_matrix[run_index]
                            metric_onset = detect_metric_onset(metric_series)
                            loss_onset = loss_onsets[run_index]
                            lead = (
                                loss_onset["task"] - metric_onset["task"]
                                if loss_onset["task"] is not None and metric_onset["task"] is not None
                                else None
                            )
                            key = (
                                experiment["experiment_id"], method_name,
                                metric_name, scope, diagnostic_mode, run_index,
                            )
                            trajectory_store[key] = metric_series
                            onset_rows.append(
                                {
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
                                    "loss_mode": loss_mode,
                                    "diagnostic_mode": diagnostic_mode,
                                    "loss_onset_task": loss_onset["task"],
                                    "loss_baseline": loss_onset["baseline"],
                                    "loss_threshold": loss_onset["threshold"],
                                    "metric_onset_task": metric_onset["task"],
                                    "metric_baseline": metric_onset["baseline"],
                                    "metric_threshold": metric_onset["threshold"],
                                    "metric_direction": metric_onset["direction"],
                                    "lead_tasks": lead,
                                }
                            )

                            for predictor_transform in ("raw", "baseline_delta"):
                                predictor = _transform(metric_series, predictor_transform)
                                for target_transform in ("raw", "baseline_delta"):
                                    target = _transform(loss_target_matrix[run_index], target_transform)
                                    if (
                                        CORRELATION_ONLY_ONSET_RUNS
                                        and loss_onset["task"] is None
                                    ):
                                        continue
                                    for lag in LAGS:
                                        x, y = _lagged_pairs(predictor, target, lag)
                                        for correlation_kind in ("pearson", "spearman"):
                                            correlation, point_count = _correlation(
                                                x, y, correlation_kind
                                            )
                                            lag_run_rows.append(
                                                {
                                                    "aggregation_level": "run",
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
                                                    "loss_mode": loss_mode,
                                                    "diagnostic_mode": diagnostic_mode,
                                                    "predictor_transform": predictor_transform,
                                                    "target_transform": target_transform,
                                                    "lag": lag,
                                                    "correlation": correlation_kind,
                                                    "correlation_value": correlation,
                                                    "point_count": point_count,
                                                }
                                            )

    return {
        "onset_rows": onset_rows,
        "lag_run_rows": lag_run_rows,
        "trajectory_store": trajectory_store,
        "loss_store": loss_store,
        "available_diagnostic_metrics": sorted(available_diagnostic_metrics),
        "fine_grained_statistics_found": fine_grained_statistics_found,
    }


def _backprop_comparison(onset_rows):
    rows = []
    controls = [method for method in PREFERRED_METHODS if method != "backprop"]
    fields = ["experiment_id", "metric", "scope", "loss_mode", "diagnostic_mode"]
    grouped = _group(onset_rows, fields)
    for key, group in grouped.items():
        by_method_run = {
            (row["method"], row["run"]): row
            for row in group
        }
        bp_rows = [row for row in group if row["method"] == "backprop"]
        for control in controls:
            paired = [
                (bp, by_method_run[(control, bp["run"])])
                for bp in bp_rows
                if (control, bp["run"]) in by_method_run
            ]
            if not paired:
                continue
            bp_leads = [float(bp["lead_tasks"]) for bp, _ in paired if bp["lead_tasks"] not in (None, "")]
            control_leads = [float(ctrl["lead_tasks"]) for _, ctrl in paired if ctrl["lead_tasks"] not in (None, "")]
            rows.append(
                {
                    "experiment": bp_rows[0]["experiment"],
                    "experiment_id": key[0],
                    "architecture": bp_rows[0]["architecture"],
                    "metric": key[1],
                    "scope": key[2],
                    "loss_mode": key[3],
                    "diagnostic_mode": key[4],
                    "comparison_method": control,
                    "paired_runs": len(paired),
                    "backprop_median_loss_onset": _median_field([bp for bp, _ in paired], "loss_onset_task"),
                    "control_median_loss_onset": _median_field([ctrl for _, ctrl in paired], "loss_onset_task"),
                    "backprop_median_metric_onset": _median_field([bp for bp, _ in paired], "metric_onset_task"),
                    "control_median_metric_onset": _median_field([ctrl for _, ctrl in paired], "metric_onset_task"),
                    "backprop_median_lead": float(np.median(bp_leads)) if bp_leads else None,
                    "control_median_lead": float(np.median(control_leads)) if control_leads else None,
                }
            )
    return rows


def _median_field(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return float(np.median(values)) if values else None


def _rank_candidates(lead_summary, lag_summary, task_count):
    ranked = []
    for lead in lead_summary:
        if lead.get("method") == BASELINE_METHOD:
            continue
        candidates = [
            row for row in lag_summary
            if row.get("method") == lead.get("method")
            and row.get("metric") == lead.get("metric")
            and row.get("scope") == lead.get("scope")
            and row.get("loss_mode") == lead.get("loss_mode")
            and row.get("diagnostic_mode") == lead.get("diagnostic_mode")
            and row.get("predictor_transform") == "baseline_delta"
            and row.get("target_transform") == "baseline_delta"
            and row.get("correlation") == "spearman"
            and row.get("median_correlation") not in (None, "")
        ]
        best = max(
            candidates,
            key=lambda row: abs(float(row["median_correlation"])),
            default=None,
        )
        median_lead = lead.get("median_lead_tasks")
        positive_lead = max(float(median_lead), 0.0) if median_lead is not None else 0.0
        lead_component = min(positive_lead / max(1, task_count), 1.0)
        correlation_component = abs(float(best["median_correlation"])) if best and best.get("median_correlation") is not None else 0.0
        run_component = float(lead.get("positive_lead_fraction") or 0.0)
        direction_component = float(lead.get("direction_consistency") or 0.0)
        architecture_component = float(lead.get("architecture_consistency") or 0.0)
        score = (
            lead_component
            * correlation_component
            * run_component
            * direction_component
            * architecture_component
        )
        row = dict(lead)
        row.update(
            {
                "best_lag": best.get("lag") if best else None,
                "best_median_spearman": best.get("median_correlation") if best else None,
                "lead_component": lead_component,
                "correlation_component": correlation_component,
                "rank_score": score,
            }
        )
        ranked.append(row)
    return sorted(ranked, key=lambda row: row["rank_score"], reverse=True)


def _make_aligned_plots(ranked, onset_rows, trajectory_store, output_dir):
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    for index, candidate in enumerate(
        tqdm(ranked[:TOP_N_PLOTS], desc="Plotting aligned trajectories", unit="plot"),
        start=1,
    ):
        matching = [
            row for row in onset_rows
            if row.get("method") == candidate.get("method")
            and row.get("metric") == candidate.get("metric")
            and row.get("scope") == candidate.get("scope")
            and row.get("loss_mode") == candidate.get("loss_mode")
            and row.get("diagnostic_mode") == candidate.get("diagnostic_mode")
            and row.get("lead_tasks") not in (None, "")
        ]
        aligned = []
        relative_tasks = np.arange(-ALIGNMENT_WINDOW, ALIGNMENT_WINDOW + 1)
        for row in matching:
            key = (
                row["experiment_id"], row["method"], row["metric"],
                row["scope"], row["diagnostic_mode"], row["run"],
            )
            series = trajectory_store.get(key)
            if series is None:
                continue
            onset = int(row["loss_onset_task"])
            baseline = np.median(series[:min(BASELINE_TASKS, len(series))])
            values = []
            for relative in relative_tasks:
                task = onset + int(relative)
                values.append(series[task] - baseline if 0 <= task < len(series) else np.nan)
            aligned.append(values)
        if not aligned:
            continue
        matrix = np.asarray(aligned, dtype=np.float64)
        median = np.full(matrix.shape[1], np.nan)
        lower = np.full(matrix.shape[1], np.nan)
        upper = np.full(matrix.shape[1], np.nan)
        for column in range(matrix.shape[1]):
            values = matrix[:, column]
            values = values[np.isfinite(values)]
            if values.size:
                median[column] = np.median(values)
                lower[column] = np.percentile(values, 25)
                upper[column] = np.percentile(values, 75)
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(relative_tasks, median, color="tab:blue")
        axis.fill_between(relative_tasks, lower, upper, color="tab:blue", alpha=0.2)
        axis.axvline(0, color="black", linestyle="--", alpha=0.6)
        axis.set_title(
            f"Aligned onset: {candidate['metric']} / {candidate['scope']} / {candidate['method']}"
        )
        axis.set_xlabel("Tasks relative to detected loss onset")
        axis.set_ylabel("Metric change from early baseline")
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"aligned_onset_{index:02d}.png"), dpi=180)
        plt.close(fig)


def _make_scatter_plots(ranked, onset_rows, trajectory_store, loss_store, output_dir):
    plot_dir = os.path.join(output_dir, "plots")
    for index, candidate in enumerate(
        tqdm(ranked[:TOP_N_PLOTS], desc="Plotting lagged scatters", unit="plot"),
        start=1,
    ):
        if candidate.get("best_lag") is None:
            continue
        lag = int(candidate["best_lag"])
        matching = [
            row for row in onset_rows
            if row.get("method") == candidate.get("method")
            and row.get("metric") == candidate.get("metric")
            and row.get("scope") == candidate.get("scope")
            and row.get("loss_mode") == candidate.get("loss_mode")
            and row.get("diagnostic_mode") == candidate.get("diagnostic_mode")
            and row.get("loss_onset_task") not in (None, "")
        ]
        x_values, y_values = [], []
        for row in matching:
            trajectory_key = (
                row["experiment_id"], row["method"], row["metric"],
                row["scope"], row["diagnostic_mode"], row["run"],
            )
            loss_key = (row["experiment_id"], row["method"], row["loss_mode"], row["run"])
            metric_series = trajectory_store.get(trajectory_key)
            loss_series = loss_store.get(loss_key)
            if metric_series is None or loss_series is None:
                continue
            predictor = _baseline_delta(metric_series)
            target = _baseline_delta(loss_series)
            x, y = _lagged_pairs(predictor, target, lag)
            x_values.extend(x.tolist())
            y_values.extend(y.tolist())
        if len(x_values) < MIN_CORRELATION_POINTS:
            continue
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(x_values, y_values, s=12, alpha=0.25)
        axis.set_title(
            f"{candidate['metric']} / {candidate['scope']} → future loss gap (lag {lag})"
        )
        axis.set_xlabel("Metric change from early baseline")
        axis.set_ylabel("Future loss change from early baseline")
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"future_loss_scatter_{index:02d}.png"), dpi=180)
        plt.close(fig)


def _make_heatmap(lag_summary, output_dir):
    rows = [
        row for row in lag_summary
        if row.get("method") == PLOT_HEATMAP_METHOD
        and row.get("predictor_transform") == "baseline_delta"
        and row.get("target_transform") == "baseline_delta"
        and row.get("correlation") == "spearman"
    ]
    if not rows:
        return
    preferred_loss = "fresh_gap" if any(row.get("loss_mode") == "fresh_gap" for row in rows) else "raw_loss"
    preferred_diag = "fresh_delta" if any(row.get("diagnostic_mode") == "fresh_delta" for row in rows) else "raw"
    rows = [row for row in rows if row.get("loss_mode") == preferred_loss and row.get("diagnostic_mode") == preferred_diag]
    if not rows:
        return
    labels = sorted({f"{row['metric']} / {row['scope']}" for row in rows})
    lags = sorted({int(row["lag"]) for row in rows})
    values = np.full((len(labels), len(lags)), np.nan)
    for row in rows:
        label_index = labels.index(f"{row['metric']} / {row['scope']}")
        lag_index = lags.index(int(row["lag"]))
        values[label_index, lag_index] = row.get("median_correlation")
    fig_height = max(4, 0.35 * len(labels) + 2)
    fig, axis = plt.subplots(figsize=(9, fig_height))
    image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_xticks(np.arange(len(lags)))
    axis.set_xticklabels(lags)
    axis.set_xlabel("Lag (metric at t-lag predicts loss at t)")
    axis.set_title(f"Median Spearman future-loss correlation ({PLOT_HEATMAP_METHOD})")
    fig.colorbar(image, ax=axis, label="Median Spearman correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plots", "spearman_lag_heatmap.png"), dpi=180)
    plt.close(fig)


def write_outputs(analysis, loaded, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("[metric predictors] Aggregating onset and lag statistics...")
    onset_rows = analysis["onset_rows"]
    lag_run_rows = analysis["lag_run_rows"]
    lead_summary = _aggregate_leads(
        onset_rows,
        ["method", "metric", "scope", "loss_mode", "diagnostic_mode"],
    )
    layer_summary = _aggregate_leads(
        onset_rows,
        ["metric", "scope", "loss_mode", "diagnostic_mode"],
    )
    architecture_rows = _aggregate_leads(
        onset_rows,
        ["architecture_key", "architecture", "method", "metric", "scope", "loss_mode", "diagnostic_mode"],
    )
    lag_summary = _aggregate_lags(lag_run_rows)
    # Use the largest observed task count for the transparent lead component.
    task_counts = []
    for experiment in loaded:
        for payload in experiment["methods"].values():
            values = _as_task_matrix(_metric_values(payload.get("results", {}), "final_loss", "global"))
            if values is not None:
                task_counts.append(values.shape[1])
    ranked = _rank_candidates(lead_summary, lag_summary, max(task_counts or [1]))

    _write_csv(os.path.join(output_dir, "metric_onset_by_run.csv"), onset_rows)
    _write_csv(
        os.path.join(output_dir, "metric_lag_correlations.csv"),
        lag_run_rows + [dict(row, aggregation_level="aggregate") for row in lag_summary],
    )
    _write_csv(os.path.join(output_dir, "metric_lead_summary.csv"), lead_summary)
    _write_csv(os.path.join(output_dir, "metric_layer_summary.csv"), layer_summary)
    _write_csv(os.path.join(output_dir, "metric_architecture_consistency.csv"), architecture_rows)
    _write_csv(os.path.join(output_dir, "backprop_comparison.csv"), _backprop_comparison(onset_rows))
    _write_csv(os.path.join(output_dir, "metric_ranked_summary.csv"), ranked)

    print("[metric predictors] Writing ranked summary and diagnostic plots...")
    with open(os.path.join(output_dir, "early_warning_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("Top candidate early indicators of plasticity loss\n")
        handle.write("=================================================\n")
        for index, row in enumerate(ranked[:10], start=1):
            handle.write(
                f"{index}. {row['metric']} / {row['scope']} / {row['method']}\n"
                f"   median lead: {row.get('median_lead_tasks')} tasks\n"
                f"   architectures consistent: {row.get('architectures_with_positive_median_lead')}"
                f"/{row.get('architectures_with_valid_runs')}\n"
                f"   runs consistent: {row.get('positive_lead_fraction')}\n"
                f"   best future-loss lag: {row.get('best_lag')} tasks\n"
                f"   median Spearman: {row.get('best_median_spearman')}\n"
                f"   transparent score: {row.get('rank_score')}\n"
            )

    _make_aligned_plots(ranked, onset_rows, analysis["trajectory_store"], output_dir)
    _make_scatter_plots(ranked, onset_rows, analysis["trajectory_store"], analysis["loss_store"], output_dir)
    _make_heatmap(lag_summary, output_dir)
    return {
        "onset_rows": onset_rows,
        "lag_run_rows": lag_run_rows,
        "lead_summary": lead_summary,
        "lag_summary": lag_summary,
        "ranked": ranked,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", action="append", dest="experiment_dirs")
    parser.add_argument("--root", action="append", dest="roots")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--no-fresh-normalization", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    experiment_dirs = discover_experiment_dirs(args.experiment_dirs, args.roots)
    if not experiment_dirs:
        raise SystemExit(
            "No experiments found. Edit EXPERIMENT_DIRS/EXPERIMENT_ROOTS near the top "
            "of analyze_metric_predictors.py or pass --experiment-dir/--root."
        )
    global USE_FRESH_NORMALIZATION
    USE_FRESH_NORMALIZATION = not args.no_fresh_normalization
    loaded = load_experiments(experiment_dirs, args.methods or METHODS_TO_ANALYZE)
    if not loaded:
        raise SystemExit("No requested method result files could be loaded.")
    analysis = analyze_loaded_experiments(loaded)
    outputs = write_outputs(analysis, loaded, args.output_dir)
    print("Top candidate early indicators of plasticity loss")
    for index, row in enumerate(outputs["ranked"][:10], start=1):
        print(
            f"{index}. {row['metric']} / {row['scope']} / {row['method']} | "
            f"median lead={row.get('median_lead_tasks')} tasks | "
            f"architectures={row.get('architectures_with_positive_median_lead')}"
            f"/{row.get('architectures_with_valid_runs')} | "
            f"runs={row.get('positive_lead_fraction')} | "
            f"best lag={row.get('best_lag')} | "
            f"median Spearman={row.get('best_median_spearman')}"
        )
    if analysis["fine_grained_statistics_found"]:
        print(
            "Note: method_statistics were present, but they are per-task intervention "
            "statistics, not finer-grained diagnostic trajectories. Analysis remains task-level."
        )
    else:
        print("Analysis resolution: task-level; no finer-grained diagnostic measurements were saved.")
    print(f"Wrote analysis outputs to {args.output_dir}")
    return outputs


if __name__ == "__main__":
    main()
