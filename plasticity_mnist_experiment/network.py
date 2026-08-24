"""MLP for flattened MNIST, with utilities for ReDo (dormant-neuron reset)
and for freezing the representation (all layers but the last) for linear
probing.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain MLP: Linear -> ReLU repeated, then a final Linear classifier.

    `self.hidden_layers[i]` produces the activations that feed hidden unit
    block i; `self.output_layer` is the last (readout) layer and is never
    touched by ReDo or by the representation-freeze helpers.
    """

    def __init__(self, input_dim=784, hidden_dims=(256, 256, 256), output_dim=10):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden_dims))
        )
        self.output_layer = nn.Linear(dims[-1], output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, return_activations=False):
        activations = []
        h = x
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
            activations.append(h)
        logits = self.output_layer(h)
        if return_activations:
            return logits, activations
        return logits

    # ---- representation freezing (experiment 5) ----------------------

    def freeze_representation(self, freeze=True):
        for layer in self.hidden_layers:
            for p in layer.parameters():
                p.requires_grad = not freeze

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---- ReDo: reset dormant neurons in every hidden layer ------------

    @torch.no_grad()
    def dormant_masks(self, activations, tau=0.025):
        """For each hidden layer, mark neurons whose mean activation
        magnitude (over the given batch) is small relative to that layer's
        average as dormant, following the ReDo score:
            score_i = mean(|a_i|) / (mean_j mean(|a_j|) + eps)
        a neuron is dormant if score_i <= tau.
        """
        masks = []
        eps = 1e-9
        for act in activations:
            per_neuron = act.abs().mean(dim=0)  # (hidden_dim,)
            layer_mean = per_neuron.mean() + eps
            score = per_neuron / layer_mean
            masks.append(score <= tau)
        return masks

    @torch.no_grad()
    def redo_reset(self, masks):
        """Reinitialize dormant neurons' incoming weights/bias and zero
        their outgoing weights, in every hidden layer (no layer is exempt).
        """
        num_reset = 0
        for i, mask in enumerate(masks):
            if mask.sum() == 0:
                continue
            incoming = self.hidden_layers[i]
            outgoing = self.hidden_layers[i + 1] if i + 1 < len(self.hidden_layers) else self.output_layer

            idx = mask.nonzero(as_tuple=True)[0]
            num_reset += idx.numel()

            fresh = nn.Linear(incoming.in_features, incoming.out_features)
            incoming.weight[idx] = fresh.weight[idx]
            incoming.bias[idx] = fresh.bias[idx]

            outgoing.weight[:, idx] = 0.0
        return num_reset
