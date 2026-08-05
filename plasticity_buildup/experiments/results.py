import numpy as np
import torch


def create_empty_results(metrics):
    return {
        name: {"global": [], "per_layer": None if config.get("type") == "loss" else {}}
        for name, config in metrics.items()
    }


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return float(value)


def store_task_results(results, losses, diagnostics, metrics):
    loss_values = {"first_loss": losses[0], "average_loss": losses.mean(), "final_loss": losses[-1]}
    for metric_name, config in metrics.items():
        if config.get("type") == "loss":
            if metric_name not in loss_values:
                raise ValueError(f"Unknown loss metric: {metric_name!r}")
            results[metric_name]["global"].append(float(loss_values[metric_name]))
            continue
        diagnostic = diagnostics[metric_name]
        results[metric_name]["global"].append(_to_float(diagnostic["global"]))
        for layer_name, value in (diagnostic.get("per_layer") or {}).items():
            results[metric_name]["per_layer"].setdefault(layer_name, []).append(_to_float(value))


def convert_results_to_numpy(results):
    for metric in results.values():
        metric["global"] = np.asarray(metric["global"], dtype=np.float64)
        if metric["per_layer"] is not None:
            metric["per_layer"] = {name: np.asarray(values, dtype=np.float64) for name, values in metric["per_layer"].items()}
    return results


def stack_runs(run_results, metrics):
    if not run_results:
        raise ValueError("run_results cannot be empty")
    stacked = {}
    for metric_name in metrics:
        stacked[metric_name] = {
            "global": np.stack([run[metric_name]["global"] for run in run_results]),
            "per_layer": None,
        }
        layer_names = sorted({
            layer_name
            for run in run_results
            for layer_name in (run[metric_name]["per_layer"] or {})
        })
        if layer_names:
            stacked[metric_name]["per_layer"] = {
                layer_name: np.stack([
                    run[metric_name]["per_layer"][layer_name] for run in run_results
                ])
                for layer_name in layer_names
            }
    return stacked


def get_layer_names(results, metrics):
    return sorted({
        layer_name
        for metric_name in metrics
        for layer_name in (results[metric_name]["per_layer"] or {})
    })
