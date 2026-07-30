from __future__ import annotations

import torch

from .random_reset import reset_hidden_layer_neurons


def reset_dormant_neurons(network, activations, threshold: float, optimizer=None) -> dict:
    """Reset hidden neurons whose mean absolute activation is below threshold.

    `activations` should be a list with one tensor per hidden layer. Each tensor
    may be either one batch of activations or a concatenation/average from a
    recent window.
    """
    reset_info = []
    for layer_index, layer_activations in enumerate(activations):
        scores = layer_activations.detach().abs()
        if scores.ndim > 1:
            scores = scores.mean(dim=0)
        indices = torch.nonzero(scores < threshold, as_tuple=False).flatten()
        reset_hidden_layer_neurons(network, layer_index, indices, optimizer)
        reset_info.append({"layer": layer_index, "neurons": indices.cpu().tolist()})
    return {"num_reset": sum(len(item["neurons"]) for item in reset_info), "details": reset_info}
