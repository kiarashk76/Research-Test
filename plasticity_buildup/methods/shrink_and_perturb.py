import math

import torch

from .base import TrainingMethod


class ShrinkAndPerturbMethod(TrainingMethod):
    """Shrink parameters and add a small fresh Linear-style perturbation per task."""

    def __init__(self, model, optimizer, config=None, initial_state=None):
        super().__init__(model, optimizer, config)
        self.intervention_count = 0

    @staticmethod
    def _sample_like(parameter, is_bias=False, fan_in=None):
        sample = torch.empty_like(parameter)
        if is_bias:
            bound = 1 / math.sqrt(max(1, fan_in or parameter.numel()))
            torch.nn.init.uniform_(sample, -bound, bound)
        else:
            torch.nn.init.kaiming_uniform_(sample, a=math.sqrt(5))
        return sample

    def prepare_for_task(self, x, y):
        shrink = float(self.config.get("shrink_factor", 0.9))
        perturb = float(self.config.get("perturb_scale", 0.01))
        with torch.no_grad():
            for layer in list(self.model.hidden_layers) + [self.model.output_layer]:
                layer.weight.mul_(shrink).add_(perturb * self._sample_like(layer.weight))
                if layer.bias is not None:
                    sample = self._sample_like(layer.bias, is_bias=True, fan_in=layer.in_features)
                    layer.bias.mul_(shrink).add_(perturb * sample)
        self.optimizer.state.clear()
        self.intervention_count += 1

    def get_statistics(self):
        return {"intervention_count": self.intervention_count}
