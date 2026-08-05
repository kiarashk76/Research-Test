import torch


class RandomSupervisedPolynomialDataset:
    def __init__(self, num_samples, input_dim, output_dim=1, include_interactions=True):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.include_interactions = include_interactions
        self.X = torch.randn(num_samples, input_dim)

    def _interaction_features(self, x):
        terms = [x[:, i] * x[:, j] for i in range(self.input_dim) for j in range(i + 1, self.input_dim)]
        return torch.stack(terms, dim=1) if terms else x.new_empty(x.shape[0], 0)

    def generate_data(self, reset_x=False, noise_std=0.0):
        if reset_x:
            self.X = torch.randn(self.num_samples, self.input_dim)
        linear_weights = torch.randn(self.input_dim, self.output_dim)
        quadratic_weights = torch.randn(self.input_dim, self.output_dim)
        y = self.X @ linear_weights + self.X.square() @ quadratic_weights
        if self.include_interactions:
            interactions = self._interaction_features(self.X)
            y = y + interactions @ torch.randn(interactions.shape[1], self.output_dim)
        y = (y + torch.randn(self.output_dim)) / y.std(dim=0, keepdim=True).clamp_min(1e-8)
        if noise_std > 0:
            y = y + noise_std * torch.randn_like(y)
        return self.X, y
