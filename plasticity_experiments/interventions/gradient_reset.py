from __future__ import annotations

import torch

from .random_reset import reset_hidden_layer_neurons


def reset_small_gradient_neurons(network, reset_fraction: float, optimizer=None, gradient_scores=None) -> dict:
    """Reset the lowest-gradient hidden neurons in each hidden layer.

    This initial version scores a neuron by the mean absolute gradient of its
    incoming weights, bias, and outgoing weights from the most recent backward
    pass. It is intentionally simple and easy to refine.
    """
    reset_info = []
    for layer_index, layer in enumerate(network.hidden_layers):
        if gradient_scores is None:
            scores = current_gradient_scores(network)[layer_index]
        else:
            scores = gradient_scores[layer_index].to(layer.weight.device)

        num_reset = int(round(layer.out_features * reset_fraction))
        if num_reset <= 0:
            indices = torch.empty(0, dtype=torch.long, device=layer.weight.device)
        else:
            indices = torch.argsort(scores)[:num_reset]
        reset_hidden_layer_neurons(network, layer_index, indices, optimizer)
        reset_info.append({"layer": layer_index, "neurons": indices.cpu().tolist()})
    return {"num_reset": sum(len(item["neurons"]) for item in reset_info), "details": reset_info}


def current_gradient_scores(network) -> list[torch.Tensor]:
    scores_by_layer = []
    for layer_index, layer in enumerate(network.hidden_layers):
        scores = torch.zeros(layer.out_features, device=layer.weight.device)
        if layer.weight.grad is not None:
            scores += layer.weight.grad.detach().abs().mean(dim=1)
        if layer.bias.grad is not None:
            scores += layer.bias.grad.detach().abs()

        next_layer = (
            network.hidden_layers[layer_index + 1]
            if layer_index + 1 < len(network.hidden_layers)
            else network.output_layer
        )
        if next_layer.weight.grad is not None:
            scores += next_layer.weight.grad.detach().abs().mean(dim=0)
        scores_by_layer.append(scores)
    return scores_by_layer
