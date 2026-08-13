import os

import matplotlib.pyplot as plt
import numpy as np

from .scales import apply_metric_y_scale


def _selected_methods(method_results, methods, selected_methods):
    names = selected_methods or list(method_results)
    return [(name, methods.get(name, {}).get("label", name), method_results[name]) for name in names if name in method_results]


def _plot_lines(axis, values, label, tasks):
    values = np.asarray(values)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    line, = axis.plot(tasks, mean, label=label)
    axis.fill_between(tasks, mean - std, mean + std, alpha=0.2)
    return line


def _finish_figure(fig, axes, legend_handles, legend_labels, title=None):
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=max(1, len(legend_labels)), bbox_to_anchor=(0.5, 0.995))
    if title:
        fig.suptitle(title, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.93))


def plot_all_global_metrics(method_results, methods, metrics, save_dir, selected_methods=None, selected_metrics=None):
    metric_names = selected_metrics or list(metrics)
    metric_names = [name for name in metric_names if name in metrics]
    if not metric_names:
        return
    method_items = _selected_methods(method_results, methods, selected_methods)
    task_count = next(iter(method_results.values()))[metric_names[0]]["global"].shape[-1]
    tasks = np.arange(task_count)
    columns = 2
    rows = int(np.ceil(len(metric_names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)
    handles = []
    labels = []
    for metric_index, metric_name in enumerate(metric_names):
        axis = axes[metric_index]
        for method_name, label, result in method_items:
            line = _plot_lines(axis, result[metric_name]["global"], label, tasks)
            if metric_index == 0:
                handles.append(line)
                labels.append(label)
        axis.set_title(metrics[metric_name]["label"])
        axis.set_ylabel(metrics[metric_name]["label"])
        apply_metric_y_scale(axis, metric_name)
    for axis in axes[len(metric_names):]:
        axis.remove()
    for metric_index, axis in enumerate(axes[:len(metric_names)]):
        if metric_index // columns == rows - 1:
            axis.set_xlabel("Task")
    _finish_figure(fig, axes[:len(metric_names)], handles, labels)
    fig.savefig(os.path.join(save_dir, "all_metrics_global.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_metrics_for_layer(layer_name, method_results, methods, metrics, save_dir, selected_methods=None, selected_metrics=None):
    metric_names = selected_metrics or list(metrics)
    available = []
    for metric_name in metric_names:
        if metric_name not in metrics:
            continue
        if any((result[metric_name]["per_layer"] or {}).get(layer_name) is not None for result in method_results.values()):
            available.append(metric_name)
    if not available:
        return
    method_items = _selected_methods(method_results, methods, selected_methods)
    task_count = next(iter(method_results.values()))[available[0]]["per_layer"][layer_name].shape[-1]
    tasks = np.arange(task_count)
    columns = 2
    rows = int(np.ceil(len(available) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)
    handles = []
    labels = []
    for metric_index, metric_name in enumerate(available):
        axis = axes[metric_index]
        for method_name, label, result in method_items:
            layer_values = (result[metric_name]["per_layer"] or {}).get(layer_name)
            if layer_values is None:
                continue
            line = _plot_lines(axis, layer_values, label, tasks)
            if metric_index == 0:
                handles.append(line)
                labels.append(label)
        axis.set_title(metrics[metric_name]["label"])
        axis.set_ylabel(metrics[metric_name]["label"])
        apply_metric_y_scale(axis, metric_name)
    for axis in axes[len(available):]:
        axis.remove()
    for metric_index, axis in enumerate(axes[:len(available)]):
        if metric_index // columns == rows - 1:
            axis.set_xlabel("Task")
    _finish_figure(fig, axes[:len(available)], handles, labels, title=f"Metrics for {layer_name}")
    fig.savefig(os.path.join(save_dir, f"all_metrics_{layer_name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
