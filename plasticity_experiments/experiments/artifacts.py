from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch import nn


def save_eval_artifact(
    *,
    config: dict,
    run_dir: str | Path,
    seed: int,
    actor: str,
    network: nn.Module,
    eval_inputs: torch.Tensor,
    task_index: int,
    step_within_task: int,
    global_step: int,
    metadata: dict[str, Any] | None = None,
    extra_state_dicts: dict[str, nn.Module] | None = None,
) -> str:
    artifact_config = config.get("artifacts", {})
    if not artifact_config.get("enabled", False):
        return ""

    artifact_dir = Path(run_dir) / "artifacts" / f"seed_{seed}" / actor
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"task_{task_index:04d}_step_{global_step:08d}.pt"

    payload: dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "actor": actor,
            "task_index": task_index,
            "step_within_task": step_within_task,
            "global_step": global_step,
            **(metadata or {}),
        }
    }

    if artifact_config.get("save_weights", True):
        payload["weights"] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in network.state_dict().items()
        }
        if extra_state_dicts:
            payload["extra_state_dicts"] = {
                state_name: {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in module.state_dict().items()
                }
                for state_name, module in extra_state_dicts.items()
            }

    if artifact_config.get("save_gradients", True):
        payload["gradients"] = {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in network.named_parameters()
        }

    if artifact_config.get("save_activations", True):
        payload["activations"] = capture_activations(
            network,
            eval_inputs,
            max_examples=artifact_config.get("max_activation_examples"),
        )

    torch.save(payload, artifact_path)
    return str(artifact_path)


def save_supervised_eval_artifact(
    *,
    config: dict,
    run_dir: str | Path,
    seed: int,
    actor: str,
    network: nn.Module,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    task_index: int,
    step_within_task: int,
    global_step: int,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, float]:
    artifact_config = config.get("artifacts", {})
    if not artifact_config.get("enabled", False):
        mse = compute_supervised_mse(
            network=network,
            eval_inputs=eval_inputs,
            eval_targets=eval_targets,
        )
        return "", mse

    mse, activations, gradients = compute_full_dataset_supervised_metrics(
        network=network,
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        save_activations=artifact_config.get("save_activations", True),
        save_gradients=artifact_config.get("save_gradients", True),
        max_activation_examples=artifact_config.get("max_activation_examples"),
    )

    artifact_dir = Path(run_dir) / "artifacts" / f"seed_{seed}" / actor
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"task_{task_index:04d}_step_{global_step:08d}.pt"

    payload: dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "actor": actor,
            "task_index": task_index,
            "step_within_task": step_within_task,
            "global_step": global_step,
            "experiment": "supervised",
            "mse": mse,
            "eval_loss": mse,
            **(metadata or {}),
        },
        "mse": mse,
    }

    if artifact_config.get("save_weights", True):
        payload["weights"] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in network.state_dict().items()
        }

    if artifact_config.get("save_gradients", True):
        payload["gradients"] = gradients

    if artifact_config.get("save_activations", True):
        payload["activations"] = activations

    torch.save(payload, artifact_path)
    return str(artifact_path), mse


def compute_supervised_mse(
    *,
    network: nn.Module,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
) -> float:
    was_training = network.training
    network.eval()
    with torch.no_grad():
        predictions = network(eval_inputs)
        mse = float(F.mse_loss(predictions, eval_targets).item())
    if was_training:
        network.train()
    return mse


def compute_full_dataset_supervised_metrics(
    *,
    network: nn.Module,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    save_activations: bool,
    save_gradients: bool,
    max_activation_examples: int | None = None,
) -> tuple[float, dict[str, Any] | None, dict[str, torch.Tensor | None] | None]:
    was_training = network.training
    network.eval()
    network.zero_grad(set_to_none=True)

    outputs, hidden_activations = network(eval_inputs, return_activations=True)
    loss = F.mse_loss(outputs, eval_targets)
    mse = float(loss.detach().item())

    gradients = None
    if save_gradients:
        loss.backward()
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in network.named_parameters()
        }

    activations = None
    if save_activations:
        activation_inputs = eval_inputs
        activation_targets = eval_targets
        activation_outputs = outputs
        activation_hidden = hidden_activations
        if max_activation_examples is not None:
            max_examples = int(max_activation_examples)
            activation_inputs = activation_inputs[:max_examples]
            activation_targets = activation_targets[:max_examples]
            activation_outputs = activation_outputs[:max_examples]
            activation_hidden = [
                activation[:max_examples]
                for activation in activation_hidden
            ]

        activations = {
            "inputs": activation_inputs.detach().cpu().clone(),
            "targets": activation_targets.detach().cpu().clone(),
            "hidden": [
                activation.detach().cpu().clone()
                for activation in activation_hidden
            ],
            "outputs": activation_outputs.detach().cpu().clone(),
        }

    if was_training:
        network.train()

    return mse, activations, gradients


def capture_activations(
    network: nn.Module,
    eval_inputs: torch.Tensor,
    max_examples: int | None = None,
) -> dict[str, Any]:
    if max_examples is not None:
        eval_inputs = eval_inputs[: int(max_examples)]

    was_training = network.training
    network.eval()
    with torch.no_grad():
        outputs, hidden_activations = network(eval_inputs, return_activations=True)
    if was_training:
        network.train()

    return {
        "inputs": eval_inputs.detach().cpu().clone(),
        "hidden": [
            activation.detach().cpu().clone()
            for activation in hidden_activations
        ],
        "outputs": outputs.detach().cpu().clone(),
    }
