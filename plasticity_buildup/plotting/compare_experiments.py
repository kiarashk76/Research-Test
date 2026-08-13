"""Compare one method across multiple saved experiments without retraining."""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..experiments.workflows import load_shared_experiment
    from ..utils.io import load_method_results
    from .scales import apply_metric_y_scale
except ImportError:  # Supports `python plotting/compare_experiments.py`.
    from experiments.workflows import load_shared_experiment
    from utils.io import load_method_results
    from plotting.scales import apply_metric_y_scale

# backprop, continual_backprop, fresh, l2_init, 
# l2, low_gradient_reset, redo, reset_optimizer, shrink_and_perturb
method = "" 
TAG=f""

# Edit these values for a standalone cross-experiment comparison.
COMPARISON_SERIES = [
    # Linear Experiments
    # {
    #     "experiment_dir": "outputs/linear_xfixed/in10_h8_out1/samples512_tasks200_epochs50/adam_lr0.01",
    #     "method_name": method,
    #     "label": f"{method}, hidden=[8]",
    # },
    # {
    #     "experiment_dir": "outputs/linear_xfixed/in10_h32_out1/samples512_tasks200_epochs50/adam_lr0.01",
    #     "method_name": method,
    #     "label": f"{method}, hidden=[32]",
    # },
    # {
    #     "experiment_dir": "outputs/linear_xfixed/in10_h64_out1/samples512_tasks200_epochs50/adam_lr0.01",
    #     "method_name": method,
    #     "label": f"{method}, hidden=[64]",
    # },
    # {
    #     "experiment_dir": "outputs/linear_xfixed/in10_h256_out1/samples512_tasks200_epochs50/adam_lr0.01",
    #     "method_name": method,
    #     "label": f"{method}, hidden=[256]",
    # },
    # {
    #     "experiment_dir": "outputs/linear_xfixed/in10_h512_out1/samples512_tasks200_epochs50/adam_lr0.01",
    #     "method_name": method,
    #     "label": f"{method}, hidden=[512]",
    # },
    
    #Non-Linear Experiments
    {
        "experiment_dir": "outputs/nonlinear_xfixed/in10_h8-8_out1/samples512_tasks200_epochs50/adam_lr0.01",
        "method_name": method,
        "label": f"{method}, hidden=[8-8]",
    },
    {
        "experiment_dir": "outputs/nonlinear_xfixed/in10_h32-32_out1/samples512_tasks200_epochs50/adam_lr0.01",
        "method_name": method,
        "label": f"{method}, hidden=[32-32]",
    },
    {
        "experiment_dir": "outputs/nonlinear_xfixed/in10_h40-40-40-40_out1/samples512_tasks200_epochs50/adam_lr0.01",
        "method_name": method,
        "label": f"{method}, hidden=[40-40-40-40]",
    },
    {
        "experiment_dir": "outputs/nonlinear_xfixed/in10_h48-48-48_out1/samples512_tasks200_epochs50/adam_lr0.01",
        "method_name": method,
        "label": f"{method}, hidden=[48-48-48]",
    },
    {
        "experiment_dir": "outputs/nonlinear_xfixed/in10_h64-64_out1/samples512_tasks200_epochs50/adam_lr0.01",
        "method_name": method,
        "label": f"{method}, hidden=[64-64]",
    },
    {
        "experiment_dir": "outputs/nonlinear_xfixed/in10_h256-256_out1/samples512_tasks200_epochs50/adam_lr0.01",
        "method_name": method,
        "label": f"{method}, hidden=[256-256]",
    },
    {
        "experiment_dir": "outputs/nonlinear_xfixed/in10_h512-512_out1/samples512_tasks200_epochs50/adam_lr0.01",
        "method_name": method,
        "label": f"{method}, hidden=[512-512]",
    },

]
METRICS_TO_PLOT = [
    "final_loss",
    "dormant_fraction",
    "gradient_norm",
    "effective_rank",
]
OUTPUT_DIR = f"outputs/comparisons/nonlinear/test/"
REQUIRE_IDENTICAL_TASKS = True
INCLUDE_PER_LAYER = True
RUN_GLOBAL_COMPARISON = False

# Inputs for the tail-loss summary plot. The experiment paths are reused from
# COMPARISON_SERIES, while this list controls which methods are loaded from
# every experiment.
TAIL_LOSS_EXPERIMENTS = [
    {
        "experiment_dir": item["experiment_dir"],
        "label": item["label"].split(", ", 1)[-1],
    }
    for item in COMPARISON_SERIES
]
TAIL_LOSS_METHODS = [
    "backprop",
    "continual_backprop",
    "fresh",
    "l2_init",
    "l2",
    "low_gradient_reset",
    "redo",
    "reset_optimizer",
    "shrink_and_perturb",
]
TAIL_LOSS_FRACTION = 0.20
TAIL_LOSS_OUTPUT_DIR = "outputs/comparisons/tail_loss_non-linear"
TAIL_LOSS_LOG_SCALE = True
RUN_TAIL_LOSS_SUMMARY = True

# Inputs for comparing layers within one experiment and method.
LAYER_COMPARISON_EXPERIMENT = COMPARISON_SERIES[0]["experiment_dir"]
LAYER_COMPARISON_METHOD = method
LAYER_COMPARISON_METRICS = ["final_loss", "dormant_fraction", "gradient_norm", "effective_rank"]
LAYER_COMPARISON_LAYERS = None  # e.g. ["hidden_0", "hidden_1"]
LAYER_COMPARISON_OUTPUT_DIR = "outputs/comparisons/layers"
RUN_LAYER_COMPARISON = False

def _load_series(series_spec):
    experiment_dir = series_spec["experiment_dir"]
    method_name = series_spec["method_name"]
    metadata = load_shared_experiment(experiment_dir)
    payloads = load_method_results(
        os.path.join(experiment_dir, "method_results"),
        [method_name],
        expected_experiment_id=metadata["experiment_id"],
    )
    payload = payloads[0]
    config = metadata["config"]
    architecture = f"hidden={config.get('hidden_dims')}"
    return {
        "label": series_spec.get("label", f"{method_name}, {architecture}"),
        "experiment_dir": experiment_dir,
        "method_name": method_name,
        "experiment_id": metadata["experiment_id"],
        "metadata": metadata,
        "payload": payload,
        "results": payload["results"],
    }


def load_comparison_series(series, require_identical_tasks=True):
    """Load labeled method results from multiple experiment directories."""
    if not series:
        raise ValueError("series cannot be empty")
    loaded = [_load_series(spec) for spec in series]

    first_metadata = loaded[0]["metadata"]
    first_runs = first_metadata["runs"]
    first_task_hashes = [run["tasks_hash"] for run in first_runs]
    for item in loaded[1:]:
        metadata = item["metadata"]
        if require_identical_tasks:
            task_hashes = [run["tasks_hash"] for run in metadata["runs"]]
            if task_hashes != first_task_hashes:
                raise ValueError(
                    "Cross-experiment comparison requires identical saved task sequences; "
                    f"they differ for {item['experiment_dir']}"
                )

    common_metrics = set(loaded[0]["results"])
    for item in loaded[1:]:
        common_metrics.intersection_update(item["results"])
    if not common_metrics:
        raise ValueError("Selected method results have no common metrics")
    return loaded, common_metrics


def _plot_series(axis, values, label, tasks):
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("Global result arrays must have shape [num_runs, num_tasks]")
    if values.shape[-1] != len(tasks):
        raise ValueError("Compared result arrays must have the same number of tasks")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    line, = axis.plot(tasks, mean, label=label)
    axis.fill_between(tasks, mean - std, mean + std, alpha=0.2)
    return line


def _finish_figure(fig, axes, handles, labels, title=None):
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), bbox_to_anchor=(0.5, 0.995))
    if title:
        fig.suptitle(title, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.93))


def _plot_global(loaded, metric_names, output_dir, tag=""):
    first_result = loaded[0]["results"]
    task_count = first_result[metric_names[0]]["global"].shape[-1]
    tasks = np.arange(task_count)
    columns = 2
    rows = int(np.ceil(len(metric_names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)
    handles, labels = [], []
    for metric_index, metric_name in enumerate(metric_names):
        axis = axes[metric_index]
        for item in loaded:
            line = _plot_series(axis, item["results"][metric_name]["global"], item["label"], tasks)
            if metric_index == 0:
                handles.append(line)
                labels.append(item["label"])
        axis.set_title(metric_name)
        axis.set_ylabel(metric_name)
        apply_metric_y_scale(axis, metric_name)
    for axis in axes[len(metric_names):]:
        axis.remove()
    for index, axis in enumerate(axes[:len(metric_names)]):
        if index // columns == rows - 1:
            axis.set_xlabel("Task")
    _finish_figure(fig, axes[:len(metric_names)], handles, labels)
    fig.savefig(os.path.join(output_dir, f"all_metrics_global_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_layers(loaded, metric_names, output_dir, tag=""):
    layer_names = sorted({
        layer_name
        for item in loaded
        for metric_name in metric_names
        for layer_name in (item["results"][metric_name]["per_layer"] or {})
    })
    for layer_name in layer_names:
        available = [
            metric_name
            for metric_name in metric_names
            if any(layer_name in (item["results"][metric_name]["per_layer"] or {}) for item in loaded)
        ]
        columns = 2
        rows = int(np.ceil(len(available) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows), sharex=True)
        axes = np.atleast_1d(axes).reshape(-1)
        handles, labels = [], []
        for metric_index, metric_name in enumerate(available):
            axis = axes[metric_index]
            for item in loaded:
                values = (item["results"][metric_name]["per_layer"] or {}).get(layer_name)
                if values is None:
                    continue
                task_count = values.shape[-1]
                line = _plot_series(axis, values, item["label"], np.arange(task_count))
                if metric_index == 0:
                    handles.append(line)
                    labels.append(item["label"])
            axis.set_title(metric_name)
            axis.set_ylabel(metric_name)
            apply_metric_y_scale(axis, metric_name)
        for axis in axes[len(available):]:
            axis.remove()
        for index, axis in enumerate(axes[:len(available)]):
            if index // columns == rows - 1:
                axis.set_xlabel("Task")
        _finish_figure(fig, axes[:len(available)], handles, labels, title=f"Metrics for {layer_name}")
        fig.savefig(os.path.join(output_dir, f"all_metrics_{layer_name}_{tag}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_cross_experiment_methods(
    series,
    output_dir,
    selected_metrics=None,
    require_identical_tasks=True,
    include_per_layer=False,
    tag="",
):
    """Plot a selected method across multiple independently saved experiments."""
    loaded, common_metrics = load_comparison_series(series, require_identical_tasks=require_identical_tasks)
    metric_names = list(selected_metrics) if selected_metrics is not None else list(common_metrics)
    missing = [name for name in metric_names if name not in common_metrics]
    if missing:
        raise ValueError(f"Metrics are not available in every comparison series: {missing}")
    if not metric_names:
        raise ValueError("No metrics selected for plotting")

    first_task_count = loaded[0]["results"][metric_names[0]]["global"].shape[-1]
    for item in loaded[1:]:
        for metric_name in metric_names:
            if item["results"][metric_name]["global"].shape[-1] != first_task_count:
                raise ValueError("Compared experiments must have the same number of tasks")

    os.makedirs(output_dir, exist_ok=True)
    _plot_global(loaded, metric_names, output_dir, tag=tag)
    if include_per_layer:
        _plot_layers(loaded, metric_names, output_dir, tag=tag)
    with open(os.path.join(output_dir, f"comparison_metadata_{tag}.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "series": [
                    {
                        "label": item["label"],
                        "experiment_dir": item["experiment_dir"],
                        "method_name": item["method_name"],
                        "experiment_id": item["experiment_id"],
                    }
                    for item in loaded
                ],
                "metrics": metric_names,
                "require_identical_tasks": require_identical_tasks,
            },
            handle,
            indent=2,
        )
    return output_dir


def _load_experiment_methods(experiment_spec, method_names):
    """Load several method result files belonging to one experiment."""
    experiment_dir = experiment_spec["experiment_dir"]
    metadata = load_shared_experiment(experiment_dir)
    try:
        payloads = load_method_results(
            os.path.join(experiment_dir, "method_results"),
            list(method_names),
            expected_experiment_id=metadata["experiment_id"],
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing method result while loading {experiment_dir}. "
            f"Generate these methods first: {list(method_names)}"
        ) from exc

    return {
        "label": experiment_spec.get("label", experiment_dir),
        "experiment_dir": experiment_dir,
        "metadata": metadata,
        "methods": {payload["method_name"]: payload["results"] for payload in payloads},
    }


def _tail_mean(values, tail_fraction):
    """Return mean and run-to-run standard deviation over the final tasks."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected result arrays with shape [num_runs, num_tasks]")
    task_count = values.shape[-1]
    tail_task_count = max(1, int(np.ceil(task_count * tail_fraction)))
    per_run_means = values[:, -tail_task_count:].mean(axis=1)
    return float(per_run_means.mean()), float(per_run_means.std()), tail_task_count


def plot_tail_loss_by_experiment(
    experiment_specs,
    method_names,
    output_dir,
    tail_fraction=0.20,
    metric_name="final_loss",
    require_identical_tasks=True,
    log_loss=True,
    tag="",
):
    """Plot final-task loss summaries for several methods and experiments.

    Each experiment gets one x-axis group. Each method is shown as a dot in
    that group,
    where the dot is the mean of the selected metric over the final
    ``tail_fraction`` of tasks, averaged across runs. Error bars show the
    standard deviation of those per-run tail means.

    ``experiment_specs`` must contain dictionaries with ``experiment_dir`` and
    optionally ``label``. ``method_names`` must be method result filenames
    present in every experiment's ``method_results`` directory.
    """
    if not experiment_specs:
        raise ValueError("experiment_specs cannot be empty")
    method_names = list(dict.fromkeys(method_names))
    if not method_names:
        raise ValueError("method_names cannot be empty")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be greater than 0 and at most 1")

    loaded = [_load_experiment_methods(spec, method_names) for spec in experiment_specs]

    first_task_hashes = [run["tasks_hash"] for run in loaded[0]["metadata"]["runs"]]
    for item in loaded[1:]:
        if require_identical_tasks:
            task_hashes = [run["tasks_hash"] for run in item["metadata"]["runs"]]
            if task_hashes != first_task_hashes:
                raise ValueError(
                    "Tail-loss comparison requires identical saved task sequences; "
                    f"they differ for {item['experiment_dir']}"
                )

    summary = {}
    all_means = []
    task_counts = set()
    tail_task_counts = set()
    for item in loaded:
        experiment_summary = {}
        for method_name in method_names:
            results = item["methods"][method_name]
            if metric_name not in results:
                raise ValueError(
                    f"Metric {metric_name!r} is not available for method {method_name!r} "
                    f"in {item['experiment_dir']}"
                )
            values = results[metric_name]["global"]
            values = np.asarray(values)
            if values.ndim != 2:
                raise ValueError(
                    f"Expected {metric_name!r} results for {method_name!r} to have "
                    "shape [num_runs, num_tasks]"
                )
            task_counts.add(values.shape[-1])
            mean, std, tail_task_count = _tail_mean(values, tail_fraction)
            experiment_summary[method_name] = {
                "mean": mean,
                "std": std,
                "tail_task_count": tail_task_count,
            }
            all_means.append(mean)
            tail_task_counts.add(tail_task_count)
        summary[item["label"]] = experiment_summary

    if len(task_counts) != 1:
        raise ValueError("Compared experiments must have the same number of tasks")
    if log_loss and any(mean <= 0 for mean in all_means):
        raise ValueError("A log-scaled loss axis requires all tail-loss means to be positive")

    os.makedirs(output_dir, exist_ok=True)
    experiment_labels = [item["label"] for item in loaded]
    x_positions = np.arange(len(experiment_labels), dtype=float)
    if len(method_names) == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.28, 0.28, len(method_names))

    fig_width = max(8.0, 1.5 * len(experiment_labels) + 3.0)
    fig, axis = plt.subplots(figsize=(fig_width, 7))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(method_names)))
    for method_index, method_name in enumerate(method_names):
        means = np.asarray([summary[label][method_name]["mean"] for label in experiment_labels])
        stds = np.asarray([summary[label][method_name]["std"] for label in experiment_labels])
        if log_loss:
            lower_errors = np.minimum(stds, means * (1.0 - 1e-9))
            yerr = np.vstack([lower_errors, stds])
        else:
            yerr = stds
        axis.errorbar(
            x_positions + offsets[method_index],
            means,
            yerr=yerr,
            fmt="o",
            markersize=7,
            capsize=3,
            linestyle="none",
            color=colors[method_index],
            label=method_name,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(experiment_labels)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    axis.set_xlabel("Experiment")
    axis.set_ylabel(
        f"Mean {metric_name} over final {tail_fraction:.0%} of tasks "
        f"({next(iter(tail_task_counts))} tasks)"
    )
    if log_loss:
        apply_metric_y_scale(axis, metric_name)
    else:
        apply_metric_y_scale(axis, metric_name, enabled=False)
    for separator in x_positions[:-1] + 0.5:
        axis.axvline(separator, color="gray", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout()

    suffix = f"_{tag}" if tag else ""
    percentage = int(round(tail_fraction * 100))
    plot_path = os.path.join(output_dir, f"tail_{metric_name}_{percentage}pct{suffix}.png")
    summary_path = os.path.join(output_dir, f"tail_{metric_name}_{percentage}pct{suffix}.json")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metric": metric_name,
                "tail_fraction": tail_fraction,
                "tail_task_count": next(iter(tail_task_counts)),
                "experiments": summary,
            },
            handle,
            indent=2,
        )
    return {"plot_path": plot_path, "summary_path": summary_path, "summary": summary}


def plot_experiment_metrics_across_layers(
    experiment_dir,
    method_name,
    metric_names,
    output_dir,
    selected_layers=None,
    log_metrics=True,
    tag="",
):
    """Plot selected metrics for several layers from one experiment/method.

    The output contains one subplot per metric. Every subplot contains one
    line per layer, with the mean across runs and a shaded run-to-run standard
    deviation band.
    """
    metric_names = list(dict.fromkeys(metric_names))
    if not metric_names:
        raise ValueError("metric_names cannot be empty")

    metadata = load_shared_experiment(experiment_dir)
    payload = load_method_results(
        os.path.join(experiment_dir, "method_results"),
        [method_name],
        expected_experiment_id=metadata["experiment_id"],
    )[0]
    results = payload["results"]
    metric_configs = payload.get("metrics", {}) or metadata.get("metrics", {})

    available_metrics = []
    all_layers = []
    for metric_name in metric_names:
        if metric_name not in results:
            raise ValueError(f"Metric {metric_name!r} is not available for {method_name!r}")
        per_layer = results[metric_name].get("per_layer") or {}
        available_metrics.append(metric_name)
        if per_layer:
            all_layers.extend(per_layer)

    if selected_layers is None:
        # Some diagnostics, including gradient_norm, also report the output
        # layer. By default this comparison is limited to hidden layers.
        layer_names = sorted({layer for layer in all_layers if layer.startswith("hidden_")})
    else:
        layer_names = list(dict.fromkeys(selected_layers))
        missing = [layer for layer in layer_names if layer not in set(all_layers)]
        if missing:
            raise ValueError(f"Requested layers are not available: {missing}")
    task_counts = set()
    for metric_name in available_metrics:
        per_layer = results[metric_name]["per_layer"] or {}
        if per_layer:
            for layer_name in layer_names:
                values = per_layer.get(layer_name)
                if values is None:
                    continue
                values = np.asarray(values)
                if values.ndim != 2:
                    raise ValueError(
                        f"Expected {metric_name!r}/{layer_name!r} results to have "
                        "shape [num_runs, num_tasks]"
                    )
                task_counts.add(values.shape[-1])
        else:
            values = np.asarray(results[metric_name]["global"])
            if values.ndim != 2:
                raise ValueError(
                    f"Expected global {metric_name!r} results to have "
                    "shape [num_runs, num_tasks]"
                )
            task_counts.add(values.shape[-1])
    if len(task_counts) != 1:
        raise ValueError("Selected layer results must have the same number of tasks")

    fig, axes = plt.subplots(
        len(available_metrics),
        1,
        figsize=(13, 4.5 * len(available_metrics)),
        sharex=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)
    handles, labels = [], []
    seen_layers = set()
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(layer_names)))

    for metric_index, metric_name in enumerate(available_metrics):
        axis = axes[metric_index]
        per_layer = results[metric_name]["per_layer"] or {}
        metric_label = metric_configs.get(metric_name, {}).get("label", metric_name)
        if per_layer:
            series = [
                (layer_name, per_layer.get(layer_name), colors[layer_index])
                for layer_index, layer_name in enumerate(layer_names)
                if per_layer.get(layer_name) is not None
            ]
        else:
            # Loss metrics are global, so show one experiment-level line.
            series = [("global", results[metric_name]["global"], "black")]

        for series_name, values, color in series:
            values = np.asarray(values, dtype=np.float64)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            tasks = np.arange(values.shape[-1])
            line, = axis.plot(tasks, mean, color=color, label=series_name)
            axis.fill_between(
                tasks,
                mean - std,
                mean + std,
                color=color,
                alpha=0.12,
            )
            if series_name not in seen_layers:
                handles.append(line)
                labels.append(series_name)
                seen_layers.add(series_name)

        axis.set_title(metric_label)
        axis.set_ylabel(metric_label)
        apply_metric_y_scale(axis, metric_name, enabled=log_metrics)
        axis.grid(alpha=0.2)

    axes[-1].set_xlabel("Task")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(labels)),
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(f"{method_name}: metrics across layers", y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    plot_path = os.path.join(output_dir, f"{method_name}_layers{suffix}.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


if __name__ == "__main__":
    if not COMPARISON_SERIES:
        raise SystemExit("Edit COMPARISON_SERIES near the top of compare_experiments.py first.")
    if RUN_GLOBAL_COMPARISON:
        plot_cross_experiment_methods(
            COMPARISON_SERIES,
            OUTPUT_DIR,
            selected_metrics=METRICS_TO_PLOT,
            require_identical_tasks=REQUIRE_IDENTICAL_TASKS,
            include_per_layer=INCLUDE_PER_LAYER,
            tag=TAG,
        )
    if RUN_TAIL_LOSS_SUMMARY:
        plot_tail_loss_by_experiment(
            TAIL_LOSS_EXPERIMENTS,
            TAIL_LOSS_METHODS,
            TAIL_LOSS_OUTPUT_DIR,
            tail_fraction=TAIL_LOSS_FRACTION,
            metric_name="final_loss",
            require_identical_tasks=REQUIRE_IDENTICAL_TASKS,
            log_loss=TAIL_LOSS_LOG_SCALE,
            tag=TAG,
        )
    if RUN_LAYER_COMPARISON:
        plot_experiment_metrics_across_layers(
            LAYER_COMPARISON_EXPERIMENT,
            LAYER_COMPARISON_METHOD,
            LAYER_COMPARISON_METRICS,
            LAYER_COMPARISON_OUTPUT_DIR,
            selected_layers=LAYER_COMPARISON_LAYERS,
            tag=TAG,
        )
