from .base import TrainingMethod


class L2RegularizationMethod(TrainingMethod):
    """Ordinary L2 regularization toward zero for the current MLP."""

    def modify_loss(self, loss, x, y):
        coefficient = float(self.config.get("coefficient", 1e-4))
        regularization = sum(parameter.square().sum() for parameter in self.model.parameters())
        return loss + coefficient * regularization

    def get_statistics(self):
        return {"coefficient": float(self.config.get("coefficient", 1e-4))}
