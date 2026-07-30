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
) -> tuple[str, float, dict[str, float]]:
    artifact_config = config.get("artifacts", {})
    analysis_config = config.get("analysis", {})
    dormant_threshold = activation_dormant_threshold(analysis_config)
    if not artifact_config.get("enabled", False):
        mse, activation_metrics = compute_supervised_mse_and_activation_metrics(
            network=network,
            eval_inputs=eval_inputs,
            eval_targets=eval_targets,
            dormant_threshold=dormant_threshold,
        )
        return "", mse, activation_metrics

    mse, activations, gradients, activation_metrics = compute_full_dataset_supervised_metrics(
        network=network,
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        save_activations=artifact_config.get("save_activations", True),
        save_gradients=artifact_config.get("save_gradients", True),
        max_activation_examples=artifact_config.get("max_activation_examples"),
        dormant_threshold=dormant_threshold,
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
        "activation_metrics": activation_metrics,
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
    return str(artifact_path), mse, activation_metrics


def compute_supervised_mse_and_activation_metrics(
    *,
    network: nn.Module,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    dormant_threshold: float,
) -> tuple[float, dict[str, float]]:
    was_training = network.training
    network.eval()
    with torch.no_grad():
        predictions, hidden_activations = network(eval_inputs, return_activations=True)
        mse = float(F.mse_loss(predictions, eval_targets).item())
        activation_metrics = compute_activation_metrics(
            hidden_activations,
            dormant_threshold=dormant_threshold,
        )
    if was_training:
        network.train()
    return mse, activation_metrics


def compute_full_dataset_supervised_metrics(
    *,
    network: nn.Module,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    save_activations: bool,
    save_gradients: bool,
    max_activation_examples: int | None = None,
    dormant_threshold: float = 0.01,
) -> tuple[
    float,
    dict[str, Any] | None,
    dict[str, torch.Tensor | None] | None,
    dict[str, float],
]:
    was_training = network.training
    network.eval()
    network.zero_grad(set_to_none=True)

    outputs, hidden_activations = network(eval_inputs, return_activations=True)
    loss = F.mse_loss(outputs, eval_targets)
    mse = float(loss.detach().item())
    activation_metrics = compute_activation_metrics(
        hidden_activations,
        dormant_threshold=dormant_threshold,
    )

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

    return mse, activations, gradients, activation_metrics


def compute_activation_metrics(
    hidden_activations: list[torch.Tensor],
    dormant_threshold: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "activation_dormant_threshold": float(dormant_threshold),
    }
    for layer_index, activation in enumerate(hidden_activations):
        values = activation.detach()
        layer_width = int(values.shape[1]) if values.ndim == 2 else int(values.numel())
        effective_rank = activation_effective_rank(values)
        normalized_effective_rank = effective_rank / max(layer_width, 1)
        mean_abs_activation = values.abs().mean(dim=0)
        active_per_neuron = values.abs().sum(dim=0)

        prefix = f"layer_{layer_index}"
        metrics[f"{prefix}_width"] = float(layer_width)
        metrics[f"{prefix}_effective_rank"] = effective_rank
        metrics[f"{prefix}_normalized_effective_rank"] = normalized_effective_rank
        metrics[f"{prefix}_dormant_fraction"] = float(
            (mean_abs_activation < dormant_threshold).float().mean().item()
        )
        metrics[f"{prefix}_zero_activation_fraction"] = float(
            (values == 0).float().mean().item()
        )
        metrics[f"{prefix}_never_active_fraction"] = float(
            (active_per_neuron == 0).float().mean().item()
        )
    return metrics


def activation_effective_rank(activation: torch.Tensor) -> float:
    centered = activation - activation.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.float())
    total = singular_values.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    entropy = -(probabilities * torch.log(probabilities)).sum()
    return float(torch.exp(entropy).item())


def activation_dormant_threshold(analysis_config: dict) -> float:
    config = analysis_config.get("dormant_neuron_fraction_by_eval_step", {})
    if isinstance(config, dict):
        return float(config.get("activation_threshold", 0.01))
    return 0.01


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
