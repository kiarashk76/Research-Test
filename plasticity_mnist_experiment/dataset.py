"""Permuted-MNIST continual-learning dataset.

Standard MNIST, but every `epochs_per_task` epochs the task switches to a new
fixed random pixel permutation (labels are untouched). Task 0 is always the
identity permutation (i.e. plain MNIST).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class PermutedMNISTTask(Dataset):
    """A single MNIST split with one fixed pixel permutation applied."""

    def __init__(self, images, labels, permutation):
        # images: (N, 784) float32 in [0, 1], shared across all tasks;
        # permutation: (784,) int64, applied lazily per-sample so tasks don't
        # each hold their own full copy of the dataset.
        self.images = images
        self.labels = labels
        self.permutation = permutation

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.images[idx, self.permutation], self.labels[idx]


class PermutedMNIST:
    """Generates the sequence of permuted-MNIST tasks.

    Args:
        num_tasks: how many distinct pixel permutations to generate.
        epochs_per_task: how many epochs each task is trained on before
            switching to the next permutation.
        seed: random seed controlling both the permutations and their order,
            so the whole task sequence is reproducible.
        data_dir: where raw MNIST is downloaded/cached.
        train: whether to build the dataset from the MNIST train or test split.
    """

    NUM_PIXELS = 28 * 28
    NUM_CLASSES = 10

    def __init__(self, num_tasks=10, epochs_per_task=5, seed=0,
                 data_dir="./data", train=True):
        self.num_tasks = num_tasks
        self.epochs_per_task = epochs_per_task
        self.seed = seed

        raw = datasets.MNIST(root=data_dir, train=train, download=True,
                              transform=transforms.ToTensor())
        images = raw.data.reshape(len(raw), -1).float() / 255.0
        labels = raw.targets.long()

        rng = np.random.RandomState(seed)
        self.permutations = [np.arange(self.NUM_PIXELS)]  # task 0: identity
        for _ in range(1, num_tasks):
            self.permutations.append(rng.permutation(self.NUM_PIXELS))

        self.tasks = [
            PermutedMNISTTask(images, labels, torch.from_numpy(perm))
            for perm in self.permutations
        ]

    def get_task(self, task_id):
        return self.tasks[task_id]

    def task_for_epoch(self, global_epoch):
        """Which task index is active at a given (0-indexed) global epoch."""
        return min(global_epoch // self.epochs_per_task, self.num_tasks - 1)

    def total_epochs(self):
        return self.num_tasks * self.epochs_per_task

    def __len__(self):
        return self.num_tasks
