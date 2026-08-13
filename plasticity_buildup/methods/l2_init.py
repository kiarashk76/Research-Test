import torch

from .base import TrainingMethod


class L2InitMethod(TrainingMethod):
    """L2 regularization toward the common initial model parameters."""

    def __init__(self, model, optimizer, config=None, initial_state=None):
        super().__init__(model, optimizer, config)
        if initial_state is None:
            raise ValueError("L2InitMethod requires the common initial model state")
        self.initial_parameters = {
            name: initial_state[name].detach().clone().to(parameter.device)
            for name, parameter in model.named_parameters()
        }

    def modify_loss(self, loss, x, y):
        coefficient = float(self.config.get("coefficient", 1e-4))
        regularization = sum(
            (parameter - self.initial_parameters[name]).square().sum()
            for name, parameter in self.model.named_parameters()
        )
        return loss + coefficient * regularization

    def get_statistics(self):
        return {"coefficient": float(self.config.get("coefficient", 1e-4))}
