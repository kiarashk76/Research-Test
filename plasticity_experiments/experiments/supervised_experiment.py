from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from interventions import (
    reset_dormant_neurons,
    reset_random_neurons,
    reset_small_gradient_neurons,
)
from interventions.gradient_reset import current_gradient_scores
from supervised_models import BasicSupervisedModel
from supervised_tasks import RandomTargetTask
from .artifacts import save_supervised_eval_artifact


def run_supervised_experiment(config: dict) -> dict:
    seed = int(config["seed"])
    torch.manual_seed(seed)

    supervised_config = config["supervised"]
    network_config = config["network"]
    intervention_config = config["intervention"]
    intervention_type = intervention_config["type"]
    output_dir = Path(config["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_interval = int(supervised_config.get("eval_interval", 1))

    task = RandomTargetTask(
        num_inputs=supervised_config["num_inputs"],
        input_dim=supervised_config["input_dim"],
        target_dim=supervised_config["target_dim"],
        num_tasks=supervised_config["num_tasks"],
        seed=seed,
    )
    task_data_path = save_supervised_task_data(config, output_dir, seed, task)

    model = make_model(config, seed_offset=0)
    rows: list[dict] = []
    global_update = 0

    for task_index in range(supervised_config["num_tasks"]):
        task.set_task(task_index)
        if intervention_type == "fresh":
            model = make_model(config, seed_offset=10_000 + task_index)
        activation_sums = None
        gradient_sums = None
        window_count = 0

        for update in range(1, supervised_config["steps_per_task"] + 1):
            global_update += 1
            x, y = task.sample_batch(supervised_config["batch_size"])

            train_metrics = model.train_step(x, y)
            activation_sums = update_activation_sums(model.network, x, activation_sums)
            gradient_sums = update_gradient_sums(model.network, gradient_sums)
            window_count += 1

            reset_count = 0
            if update % intervention_config["interval"] == 0:
                reset_count = apply_intervention(
                    model,
                    intervention_config,
                    activation_sums,
                    gradient_sums,
                    window_count,
                )
                activation_sums = None
                gradient_sums = None
                window_count = 0

            is_eval_step = (
                update % eval_interval == 0
                or update == supervised_config["steps_per_task"]
            )
            if not is_eval_step:
                continue

            x_eval, y_eval = task.get_evaluation_data()
            artifact_path, mse = save_supervised_eval_artifact(
                config=config,
                run_dir=output_dir,
                seed=seed,
                actor=intervention_type,
                network=model.network,
                eval_inputs=x_eval,
                eval_targets=y_eval,
                task_index=task_index,
                step_within_task=update,
                global_step=global_update,
                metadata={
                    "train_loss": train_metrics["train_loss"],
                    "intervention_type": intervention_type,
                    "num_neurons_reset": reset_count,
                },
            )
            rows.append(
                {
                    "seed": seed,
                    "task_index": task_index,
                    "update_within_task": update,
                    "global_update": global_update,
                    "model": intervention_type,
                    "train_loss": train_metrics["train_loss"],
                    "eval_loss": mse,
                    "intervention_type": intervention_type,
                    "num_neurons_reset": reset_count,
                    "artifact_path": artifact_path,
                }
            )

    csv_path = output_dir / f"seed_{seed}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"csv_path": str(csv_path), "num_rows": len(rows), "task_data_path": task_data_path}


def make_model(config: dict, seed_offset: int) -> BasicSupervisedModel:
    supervised_config = config["supervised"]
    network_config = config["network"]
    return BasicSupervisedModel(
        input_dim=supervised_config["input_dim"],
        target_dim=supervised_config["target_dim"],
        hidden_sizes=list(network_config["hidden_sizes"]),
        activation=network_config["activation"],
        learning_rate=supervised_config["learning_rate"],
        seed=int(config["seed"]) + seed_offset,
    )


def save_supervised_task_data(
    config: dict,
    output_dir: Path,
    seed: int,
    task: RandomTargetTask,
) -> str:
    task_data_config = config.get("task_data", {})
    if not task_data_config.get("enabled", False):
        return ""

    task_data_dir = output_dir / "task_data" / f"seed_{seed}"
    task_data_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "seed": seed,
        "num_inputs": task.num_inputs,
        "input_dim": task.input_dim,
        "target_dim": task.target_dim,
        "num_tasks": task.num_tasks,
        "description": (
            "Supervised task data. Inputs are shared across tasks; each task has "
            "its own target tensor for those same inputs."
        ),
    }
    torch.save(metadata, task_data_dir / "metadata.pt")

    if task_data_config.get("save_inputs", True):
        torch.save(task.x.detach().cpu().clone(), task_data_dir / "inputs.pt")

    if task_data_config.get("save_targets", True):
        targets_dir = task_data_dir / "targets"
        targets_dir.mkdir(exist_ok=True)
        for task_index, targets in enumerate(task.targets):
            target_path = targets_dir / f"task_{task_index:04d}_targets.pt"
            torch.save(targets.detach().cpu().clone(), target_path)

    return str(task_data_dir)


def update_activation_sums(network, x: torch.Tensor, activation_sums):
    with torch.no_grad():
        _, activations = network(x, return_activations=True)
    batch_scores = [activation.detach().abs().mean(dim=0) for activation in activations]
    if activation_sums is None:
        return [score.clone() for score in batch_scores]
    return [old + score for old, score in zip(activation_sums, batch_scores)]


def update_gradient_sums(network, gradient_sums):
    scores = current_gradient_scores(network)
    if gradient_sums is None:
        return [score.clone() for score in scores]
    return [old + score for old, score in zip(gradient_sums, scores)]


def apply_intervention(
    model: BasicSupervisedModel,
    intervention_config: dict,
    activation_sums,
    gradient_sums,
    window_count: int,
) -> int:
    intervention_type = intervention_config["type"]
    if intervention_type in {"none", "fresh"}:
        return 0
    if intervention_type == "random":
        info = reset_random_neurons(
            model.network,
            intervention_config["reset_fraction"],
            optimizer=model.optimizer,
        )
        return int(info["num_reset"])
    if intervention_type == "dormant":
        if activation_sums is None:
            return 0
        activation_means = [value / max(window_count, 1) for value in activation_sums]
        info = reset_dormant_neurons(
            model.network,
            activation_means,
            intervention_config["dormant_threshold"],
            optimizer=model.optimizer,
        )
        return int(info["num_reset"])
    if intervention_type == "small_gradient":
        if gradient_sums is None:
            return 0
        gradient_means = [value / max(window_count, 1) for value in gradient_sums]
        info = reset_small_gradient_neurons(
            model.network,
            intervention_config["reset_fraction"],
            optimizer=model.optimizer,
            gradient_scores=gradient_means,
        )
        return int(info["num_reset"])
    raise ValueError(f"Unknown intervention type: {intervention_type}")
