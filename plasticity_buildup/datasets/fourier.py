import math

import torch


class RandomSupervisedFourierDataset:
    def __init__(self, num_samples, input_dim, num_frequencies=32, output_dim=1, frequency_scale=1.0):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = output_dim
        self.frequency_scale = frequency_scale
        self.X = torch.randn(num_samples, input_dim)

    def generate_data(self, reset_x=False, noise_std=0.0):
        if reset_x:
            self.X = torch.randn(self.num_samples, self.input_dim)
        frequencies = self.frequency_scale * torch.randn(self.input_dim, self.num_frequencies)
        phases = 2 * math.pi * torch.rand(self.num_frequencies)
        output_weights = torch.randn(self.num_frequencies, self.output_dim) / math.sqrt(self.num_frequencies)
        features = torch.sin(self.X @ frequencies + phases)
        y = features @ output_weights + torch.randn(self.output_dim)
        if noise_std > 0:
            y = y + noise_std * torch.randn_like(y)
        return self.X, y
