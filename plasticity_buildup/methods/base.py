class TrainingMethod:
    """Hook interface for methods that modify the generic training loop."""

    def __init__(self, model, optimizer, config=None, initial_state=None):
        self.model = model
        self.optimizer = optimizer
        self.config = config or {}
        self.num_updates = 0

    def prepare_for_task(self, x, y):
        """Apply interventions that happen once before a task starts."""
        pass

    def before_task(self, x, y):
        pass

    def before_update(self, x, y):
        pass

    def modify_loss(self, loss, x, y):
        """Allow regularization methods to extend the task loss."""
        return loss

    def after_backward(self, x, y):
        """Hook called after backward and before optimizer.step()."""
        pass

    def after_update(self, x, y):
        self.num_updates += 1

    def after_task(self, x, y):
        pass

    def get_statistics(self):
        return {}
