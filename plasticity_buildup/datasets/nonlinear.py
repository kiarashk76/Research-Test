import math

import torch
import torch.nn.functional as F


class RandomSupervisedNonlinearDataset:
    def __init__(self, num_samples, input_dim, teacher_hidden_dim=32, output_dim=1, activation="relu"):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.teacher_hidden_dim = teacher_hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.X = torch.randn(num_samples, input_dim)

    def _activation(self, x):
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "gelu":
            return F.gelu(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def generate_data(self, reset_x=False, noise_std=0.0):
        if reset_x:
            self.X = torch.randn(self.num_samples, self.input_dim)
        w1 = torch.randn(self.input_dim, self.teacher_hidden_dim) / math.sqrt(self.input_dim)
        b1 = torch.randn(self.teacher_hidden_dim)
        w2 = torch.randn(self.teacher_hidden_dim, self.output_dim) / math.sqrt(self.teacher_hidden_dim)
        b2 = torch.randn(self.output_dim)
        hidden = self._activation(self.X @ w1 + b1)
        y = hidden @ w2 + b2
        if noise_std > 0:
            y = y + noise_std * torch.randn_like(y)
        return self.X, y
