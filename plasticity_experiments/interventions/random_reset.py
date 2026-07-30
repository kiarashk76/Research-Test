from __future__ import annotations

import math

import torch
from torch import nn


def reset_random_neurons(network, reset_fraction: float, optimizer=None) -> dict:
    """Randomly reset a fraction of hidden neurons in each hidden layer."""
    reset_info = []
    for layer_index, layer in enumerate(network.hidden_layers):
        num_neurons = layer.out_features
        num_reset = int(round(num_neurons * reset_fraction))
        if num_reset <= 0:
            reset_info.append({"layer": layer_index, "neurons": []})
            continue
        indices = torch.randperm(num_neurons, device=layer.weight.device)[:num_reset]
        reset_hidden_layer_neurons(network, layer_index, indices, optimizer)
        reset_info.append({"layer": layer_index, "neurons": indices.cpu().tolist()})
    return {"num_reset": sum(len(item["neurons"]) for item in reset_info), "details": reset_info}


def reset_hidden_layer_neurons(network, layer_index: int, indices: torch.Tensor, optimizer=None) -> None:
    """Reset incoming weights and outgoing connections for selected hidden units."""
    if indices.numel() == 0:
        return

    hidden_layer = network.hidden_layers[layer_index]
    next_layer = (
        network.hidden_layers[layer_index + 1]
        if layer_index + 1 < len(network.hidden_layers)
        else network.output_layer
    )

    with torch.no_grad():
        incoming_bound = 1.0 / math.sqrt(hidden_layer.in_features)
        hidden_layer.weight[indices].uniform_(-incoming_bound, incoming_bound)
        hidden_layer.bias[indices].uniform_(-incoming_bound, incoming_bound)

        outgoing_bound = 1.0 / math.sqrt(next_layer.in_features)
        next_layer.weight[:, indices].uniform_(-outgoing_bound, outgoing_bound)

    if optimizer is not None:
        clear_optimizer_state(optimizer, hidden_layer.weight, row_indices=indices)
        clear_optimizer_state(optimizer, hidden_layer.bias, row_indices=indices)
        clear_optimizer_state(optimizer, next_layer.weight, col_indices=indices)


def clear_optimizer_state(
    optimizer,
    parameter: nn.Parameter,
    row_indices: torch.Tensor | None = None,
    col_indices: torch.Tensor | None = None,
) -> None:
    state = optimizer.state.get(parameter)
    if not state:
        return
    for value in state.values():
        if not torch.is_tensor(value) or value.shape != parameter.shape:
            continue
        if row_indices is not None:
            value[row_indices] = 0
        if col_indices is not None:
            value[:, col_indices] = 0
