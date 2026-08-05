class TrainingMethod:
    """Hook interface for methods that modify the generic training loop."""

    def __init__(self, model, optimizer, config=None):
        self.model = model
        self.optimizer = optimizer
        self.config = config or {}
        self.num_updates = 0

    def before_task(self, x, y):
        pass

    def before_update(self, x, y):
        pass

    def after_update(self, x, y):
        self.num_updates += 1

    def after_task(self, x, y):
        pass

