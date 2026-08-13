import torch

from .random_reset import RandomResetMethod
from .reset_utils import all_hidden_neurons, number_to_reset


class ReDoMethod(RandomResetMethod):
    """Periodically reset dormant post-activation hidden neurons."""

    def after_update(self, x, y):
        if self._intervention_is_due():
            with torch.no_grad():
                features = self.model.hidden_features(x)
                dormant = []
                threshold = float(self.config.get("dormancy_threshold", 0.01))
                for layer_index, hidden in enumerate(features):
                    activity = hidden.abs().mean(dim=0)
                    normalized = activity / activity.mean().clamp_min(1e-12)
                    dormant.extend(
                        (layer_index, neuron_index)
                        for neuron_index in torch.where(normalized <= threshold)[0].tolist()
                    )
            self._reset_neurons(dormant)
        super(RandomResetMethod, self).after_update(x, y)
