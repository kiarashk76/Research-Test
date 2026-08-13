import torch

from .random_reset import RandomResetMethod
from .reset_utils import all_hidden_neurons, number_to_reset


class LowGradientResetMethod(RandomResetMethod):
    """Reset hidden neurons with the smallest incoming-gradient scores."""

    def __init__(self, model, optimizer, config=None, initial_state=None):
        super().__init__(model, optimizer, config, initial_state)
        self._pending_neurons = []

    def before_task(self, x, y):
        super().before_task(x, y)
        self._pending_neurons = []

    def after_backward(self, x, y):
        if not self._intervention_is_due():
            return
        neurons = all_hidden_neurons(self.model)
        scores = []
        for layer_index, neuron_index in neurons:
            layer = self.model.hidden_layers[layer_index]
            weight_gradient = layer.weight.grad
            bias_gradient = layer.bias.grad if layer.bias is not None else None
            score = torch.zeros((), device=layer.weight.device)
            if weight_gradient is not None:
                score = score + weight_gradient[neuron_index].square().sum()
            if bias_gradient is not None:
                score = score + bias_gradient[neuron_index].square()
            scores.append(score.sqrt().detach().cpu())
        count = number_to_reset(len(neurons), float(self.config.get("reset_fraction", 0.01)))
        if count:
            order = torch.argsort(torch.stack(scores))[:count].tolist()
            self._pending_neurons = [neurons[index] for index in order]

    def after_update(self, x, y):
        if self._pending_neurons:
            self._reset_neurons(self._pending_neurons)
            self._pending_neurons = []
        super(RandomResetMethod, self).after_update(x, y)
