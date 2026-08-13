import torch

from .base import TrainingMethod
from .reset_utils import all_hidden_neurons, reset_and_clear_neuron


class ContinualBackpropMethod(TrainingMethod):
    """Simple supervised-MLP continual backprop with utility-based replacement."""

    def __init__(self, model, optimizer, config=None, initial_state=None):
        super().__init__(model, optimizer, config)
        self.ages = [torch.zeros(layer.out_features, dtype=torch.long) for layer in model.hidden_layers]
        self.utilities = [torch.zeros(layer.out_features) for layer in model.hidden_layers]
        self.utility_updates = 0
        self.replacement_budget = 0.0
        self.total_reset_count = 0
        self.reset_count_this_task = 0

    def before_task(self, x, y):
        self.reset_count_this_task = 0

    def after_update(self, x, y):
        with torch.no_grad():
            features = self.model.hidden_features(x)
            self.utility_updates += 1
            decay = float(self.config.get("utility_decay", 0.99))
            for layer_index, hidden in enumerate(features):
                next_layer = self.model.get_next_layer(layer_index)
                outgoing_magnitude = next_layer.weight.detach().abs().mean(dim=0)
                contribution = hidden.detach().abs().mean(dim=0).cpu() * outgoing_magnitude.cpu()
                self.utilities[layer_index].mul_(decay).add_(contribution, alpha=1.0 - decay)
                self.ages[layer_index] += 1

            total_neurons = sum(age.numel() for age in self.ages)
            self.replacement_budget += float(self.config.get("replacement_rate", 0.001)) * total_neurons
            replacement_count = int(self.replacement_budget)
            mature_threshold = int(self.config.get("maturity_threshold", 100))
            mature = [
                (
                    layer_index,
                    neuron_index,
                    (
                        self.utilities[layer_index][neuron_index]
                        / max(1e-12, 1.0 - decay ** self.utility_updates)
                    ).item(),
                )
                for layer_index, age in enumerate(self.ages)
                for neuron_index in range(age.numel())
                if age[neuron_index] >= mature_threshold
            ]
            replacement_count = min(replacement_count, len(mature))
            if replacement_count:
                mature.sort(key=lambda item: item[2])
                selected = mature[:replacement_count]
                for layer_index, neuron_index, _ in selected:
                    reset_and_clear_neuron(self.model, self.optimizer, layer_index, neuron_index)
                    self.ages[layer_index][neuron_index] = 0
                    self.utilities[layer_index][neuron_index] = 0.0
                self.replacement_budget -= replacement_count
                self.total_reset_count += replacement_count
                self.reset_count_this_task += replacement_count
        super().after_update(x, y)

    def get_statistics(self):
        ages = torch.cat(self.ages).float()
        decay = float(self.config.get("utility_decay", 0.99))
        correction = max(1e-12, 1.0 - decay ** self.utility_updates)
        utilities = torch.cat(self.utilities) / correction
        return {
            "total_reset_count": self.total_reset_count,
            "reset_count_this_task": self.reset_count_this_task,
            "reset_fraction_this_task": self.reset_count_this_task / max(1, len(all_hidden_neurons(self.model))),
            "mean_neuron_age": ages.mean().item(),
            "mean_utility": utilities.mean().item(),
        }
