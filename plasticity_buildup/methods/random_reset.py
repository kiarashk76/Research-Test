import torch

from .base import TrainingMethod
from .reset_utils import all_hidden_neurons, number_to_reset, reset_and_clear_neuron


class RandomResetMethod(TrainingMethod):
    """Periodically reinitialize a uniform random fraction of hidden neurons."""

    def __init__(self, model, optimizer, config=None, initial_state=None):
        super().__init__(model, optimizer, config)
        self.total_reset_count = 0
        self.reset_count_this_task = 0

    def before_task(self, x, y):
        self.reset_count_this_task = 0

    def _intervention_is_due(self):
        frequency = int(self.config.get("reset_frequency", 10))
        if frequency <= 0:
            raise ValueError("reset_frequency must be positive")
        return (self.num_updates + 1) % frequency == 0

    def _reset_neurons(self, neurons):
        for layer_index, neuron_index in neurons:
            reset_and_clear_neuron(self.model, self.optimizer, layer_index, neuron_index)
        count = len(neurons)
        self.total_reset_count += count
        self.reset_count_this_task += count

    def after_update(self, x, y):
        if self._intervention_is_due():
            neurons = all_hidden_neurons(self.model)
            count = number_to_reset(len(neurons), float(self.config.get("reset_fraction", 0.01)))
            if count:
                selected = torch.randperm(len(neurons))[:count].tolist()
                self._reset_neurons([neurons[index] for index in selected])
        super().after_update(x, y)

    def get_statistics(self):
        total = max(1, len(all_hidden_neurons(self.model)))
        return {
            "total_reset_count": self.total_reset_count,
            "reset_count_this_task": self.reset_count_this_task,
            "reset_fraction_this_task": self.reset_count_this_task / total,
        }
