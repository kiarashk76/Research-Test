import torch


class RandomSupervisedLinearDataset:
    def __init__(self, num_samples, input_dim, output_dim=1):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X = torch.randn(num_samples, input_dim)

    def generate_data(self, reset_x=False, noise_std=0.0):
        if reset_x:
            self.X = torch.randn(self.num_samples, self.input_dim)
        weights = torch.randn(self.input_dim, self.output_dim)
        bias = torch.randn(self.output_dim)
        y = self.X @ weights + bias
        if noise_std > 0:
            y = y + noise_std * torch.randn_like(y)
        return self.X, y
