import copy
import hashlib
import io
import json
import os

import torch

try:
    from ..config import METHODS as DEFAULT_METHODS, METRICS as DEFAULT_METRICS
    from ..plotting.plots import plot_all_global_metrics, plot_all_metrics_for_layer
    from ..utils.io import load_method_results, save_config, save_method_result
    from ..utils.naming import make_experiment_directory
    from .results import stack_runs
    from .runner import generate_tasks, make_model, run_method_on_tasks
except ImportError:  # Supports direct execution from plasticity_buildup/.
    from config import METHODS as DEFAULT_METHODS, METRICS as DEFAULT_METRICS
    from plotting.plots import plot_all_global_metrics, plot_all_metrics_for_layer
    from utils.io import load_method_results, save_config, save_method_result
    from utils.naming import make_experiment_directory
    from experiments.results import stack_runs
    from experiments.runner import generate_tasks, make_model, run_method_on_tasks


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _experiment_id(config, metrics, run_records):
    identity = {
        "config": config,
        "metrics": metrics,
        "loss_function": config.get("loss_function", "mse"),
        "runs": [
            {
                "seed": record["seed"],
                "tasks_hash": record["tasks_hash"],
                "initial_model_hash": record["initial_model_hash"],
            }
            for record in run_records
        ],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def create_shared_experiment(config, metrics=None, experiment_dir=None):
    """Generate and save the immutable per-run data used by all methods."""
    metrics = metrics or DEFAULT_METRICS
    experiment_dir = experiment_dir or make_experiment_directory(config)
    os.makedirs(experiment_dir, exist_ok=True)
    runs_dir = os.path.join(experiment_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    records = []

    for run_index in range(config["num_runs"]):
        run_seed = config["seed"] + run_index
        run_dir = os.path.join(runs_dir, f"run_{run_index:03d}")
        os.makedirs(run_dir, exist_ok=True)
        tasks = generate_tasks(config, run_seed)
        # The initial state is generated from the same seeded run after task creation.
        initial_model = make_model(config)
        initial_state = copy.deepcopy(initial_model.state_dict())
        task_path = os.path.join(run_dir, "tasks.pt")
        initial_path = os.path.join(run_dir, "initial_model_state.pt")
        torch.save({
            "tasks": tasks,
            "dataset_name": config["dataset"],
            "reset_x": config["reset_x"],
            "run_seed": run_seed,
        }, task_path)
        torch.save(initial_state, initial_path)
        records.append({
            "run_index": run_index,
            "run_name": f"run_{run_index:03d}",
            "seed": run_seed,
            "tasks_file": os.path.relpath(task_path, experiment_dir),
            "initial_model_file": os.path.relpath(initial_path, experiment_dir),
            "tasks_hash": _file_hash(task_path),
            "initial_model_hash": _file_hash(initial_path),
        })

    experiment_id = _experiment_id(config, metrics, records)
    metadata = {
        "experiment_id": experiment_id,
        "config": config,
        "metrics": metrics,
        "loss_function": config.get("loss_function", "mse"),
        "runs": records,
    }
    save_config(os.path.join(experiment_dir, "metadata.json"), metadata)
    return experiment_dir, metadata


def load_shared_experiment(experiment_dir):
    metadata = _load_json(os.path.join(experiment_dir, "metadata.json"))
    for record in metadata["runs"]:
        tasks_path = os.path.join(experiment_dir, record["tasks_file"])
        initial_path = os.path.join(experiment_dir, record["initial_model_file"])
        if _file_hash(tasks_path) != record["tasks_hash"] or _file_hash(initial_path) != record["initial_model_hash"]:
            raise ValueError(f"Saved run artifacts do not match experiment metadata: {record['run_name']}")
    return metadata


def run_selected_methods(experiment_dir, methods_to_run, methods=None, metrics=None, overwrite=False):
    """Load saved runs and save only the requested method result files."""
    metadata = load_shared_experiment(experiment_dir)
    config = metadata["config"]
    methods = methods or DEFAULT_METHODS
    metrics = metrics or metadata["metrics"]
    method_dir = os.path.join(experiment_dir, "method_results")
    os.makedirs(method_dir, exist_ok=True)
    saved = {}
    for method_name in methods_to_run:
        if method_name not in methods:
            raise ValueError(f"Unknown method {method_name!r}")
        result_path = os.path.join(method_dir, f"{method_name}.pkl")
        if os.path.exists(result_path) and not overwrite:
            with open(result_path, "rb") as handle:
                import pickle
                saved[method_name] = pickle.load(handle)
            continue

        per_run_results = []
        per_run_statistics = []
        method_config = methods[method_name]
        for record in metadata["runs"]:
            tasks_path = os.path.join(experiment_dir, record["tasks_file"])
            initial_path = os.path.join(experiment_dir, record["initial_model_file"])
            task_payload = torch.load(tasks_path, map_location="cpu", weights_only=False)
            initial_state = torch.load(initial_path, map_location="cpu", weights_only=False)
            run_result, run_statistics = run_method_on_tasks(
                config,
                method_name,
                method_config,
                metrics,
                task_payload["tasks"],
                initial_state,
                record["seed"],
                return_statistics=True,
            )
            per_run_results.append(run_result)
            per_run_statistics.append(run_statistics)
        stacked = stack_runs(per_run_results, metrics)
        save_method_result(
            result_path,
            metadata["experiment_id"],
            method_name,
            method_config,
            stacked,
            metrics,
            config,
            method_statistics=per_run_statistics,
        )
        with open(result_path, "rb") as handle:
            import pickle
            saved[method_name] = pickle.load(handle)
    return saved


def plot_selected_methods(experiment_dir, methods_to_plot, metrics_to_plot=None):
    """Load selected method files, verify provenance, and create plots only."""
    metadata = load_shared_experiment(experiment_dir)
    method_dir = os.path.join(experiment_dir, "method_results")
    payloads = load_method_results(method_dir, methods_to_plot, metadata["experiment_id"])
    method_results = {payload["method_name"]: payload["results"] for payload in payloads}
    methods = {payload["method_name"]: payload["method_config"] for payload in payloads}
    metrics = metadata["metrics"]
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_all_global_metrics(method_results, methods, metrics, plots_dir, selected_metrics=metrics_to_plot)
    layer_names = sorted({
        layer_name
        for result in method_results.values()
        for metric in result.values()
        for layer_name in (metric["per_layer"] or {})
    })
    for layer_name in layer_names:
        plot_all_metrics_for_layer(layer_name, method_results, methods, metrics, plots_dir, selected_metrics=metrics_to_plot)
    return plots_dir
