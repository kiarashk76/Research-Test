import json
import os
import pickle

import torch


def save_tasks(path, tasks, dataset_name, reset_x, task_generation_seed, task_generation_seeds=None):
    payload = {
        "tasks": tasks[0] if isinstance(tasks, list) and tasks and isinstance(tasks[0], list) else tasks,
        "task_runs": tasks if isinstance(tasks, list) and tasks and isinstance(tasks[0], list) else [tasks],
        "dataset_name": dataset_name,
        "reset_x": reset_x,
        "task_generation_seed": task_generation_seed,
        "task_generation_seeds": task_generation_seeds,
    }
    torch.save(payload, path)


def load_tasks(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save_initial_model_state(path, state_dict):
    torch.save(state_dict, path)


def save_results(path, results, experiment_config, methods, metrics, task_file, initial_model_file, metadata):
    payload = {
        "results": results,
        "config": experiment_config,
        "methods": methods,
        "metrics": metrics,
        "task_file": task_file,
        "initial_model_file": initial_model_file,
        "metadata": metadata,
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def save_method_result(path, experiment_id, method_name, method_config, results, metrics, config, method_statistics=None):
    payload = {
        "experiment_id": experiment_id,
        "method_name": method_name,
        "method_config": method_config,
        "results": results,
        "metrics": metrics,
        "config": config,
        "method_statistics": method_statistics or [],
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def load_results(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def combine_result_payloads(payloads):
    """Combine separately generated method payloads with shared provenance."""
    if not payloads:
        raise ValueError("payloads cannot be empty")
    identifiers = {
        payload.get("experiment_id", payload.get("metadata", {}).get("identifier"))
        for payload in payloads
    }
    if len(identifiers) != 1:
        raise ValueError("Results do not share the same task and initialization metadata")
    combined = dict(payloads[0])
    combined["results"] = {}
    combined["methods"] = {}
    for payload in payloads:
        combined["results"].update(payload["results"])
        methods = payload.get("methods")
        if methods is None:
            methods = {payload["method_name"]: payload["method_config"]}
        combined["methods"].update(methods)
    return combined


def load_method_results(directory, method_names, expected_experiment_id=None):
    payloads = []
    for method_name in method_names:
        path = os.path.join(directory, f"{method_name}.pkl")
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("method_name") != method_name:
            raise ValueError(f"Method result filename and payload disagree: {path}")
        payloads.append(payload)

    identifiers = {payload.get("experiment_id") for payload in payloads}
    if len(identifiers) != 1:
        raise ValueError("Selected method results do not share the same experiment_id")
    experiment_id = identifiers.pop()
    if expected_experiment_id is not None and experiment_id != expected_experiment_id:
        raise ValueError("Selected method results do not belong to this experiment")
    return payloads


def save_config(path, config):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
