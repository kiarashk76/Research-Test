import math

import torch
import torch.nn.functional as F


class Network(torch.nn.Module):
    """Small ReLU MLP used by the continual-learning experiments."""

    def __init__(self, input_dim=10, hidden_dims=None, output_dim=1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        layer_dims = [input_dim] + self.hidden_dims
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(self.hidden_dims))]
        )
        self.output_layer = torch.nn.Linear(self.hidden_dims[-1], output_dim)

    def hidden_features(self, x):
        features = []
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            features.append(x)
        return features

    def forward(self, x):
        return self.output_layer(self.hidden_features(x)[-1])

    @torch.no_grad()
    def save_reference_parameters(self):
        return {name: parameter.detach().clone() for name, parameter in self.named_parameters()}

    @torch.no_grad()
    def extract_features(self, x):
        return [feature.detach().clone() for feature in self.hidden_features(x)]

    @torch.no_grad()
    def reset_hidden_neuron(self, layer_index, neuron_index, zero_outgoing=True):
        """Reinitialize one hidden unit and optionally clear its outgoing weights."""
        layer = self.hidden_layers[layer_index]
        if not 0 <= neuron_index < layer.out_features:
            raise IndexError("neuron_index is outside the selected layer")

        torch.nn.init.kaiming_uniform_(layer.weight[neuron_index : neuron_index + 1], a=math.sqrt(5))
        if layer.bias is not None:
            bound = 1 / math.sqrt(layer.in_features)
            torch.nn.init.uniform_(layer.bias[neuron_index : neuron_index + 1], -bound, bound)

        if zero_outgoing:
            next_layer = self.get_next_layer(layer_index)
            if next_layer is not None:
                next_layer.weight[:, neuron_index].zero_()

    def get_next_layer(self, layer_index):
        if layer_index < len(self.hidden_layers) - 1:
            return self.hidden_layers[layer_index + 1]
        return self.output_layer
