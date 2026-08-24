"""Shared training/eval utilities and experiment runners for the Permuted
MNIST plasticity experiments (exp1.py, exp2.py, exp3.py, ...).
"""

import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from network import MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUTS_DIR = "outputs"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, dataset, batch_size=512):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total


def make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr)


def train_one_epoch(model, train_dataset, optimizer, batch_size=128, desc=None):
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.train()
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.3f}")


def redo_pass(model, train_dataset, tau=0.025, batch_size=512, layers=None):
    """Estimate dormant neurons from one batch and reset them.

    `layers`: which hidden-layer indices ReDo is allowed to reset (None means
    every hidden layer). Dormant neurons in any other layer are left alone.
    """
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    x, _ = next(iter(loader))
    x = x.to(DEVICE)
    model.eval()
    with torch.no_grad():
        _, activations = model(x, return_activations=True)
    masks = model.dormant_masks(activations, tau=tau)
    if layers is not None:
        allowed = set(layers)
        masks = [mask if i in allowed else torch.zeros_like(mask)
                 for i, mask in enumerate(masks)]
    num_reset = model.redo_reset(masks)
    model.train()
    return num_reset


# --------------------------------------------------------------------------
# Dormant-neuron fraction (per hidden layer), averaged over a full dataset
# --------------------------------------------------------------------------

@torch.no_grad()
def compute_dormant_fractions(model, dataset, batch_size=512, tau=0.025):
    """Fraction of dormant neurons in each hidden layer, following:
        a_i = mean over the dataset of |activation_i|
        s_i = a_i / mean_j(a_j)   (normalized within the layer)
        dormant if s_i <= tau
    Returns a list of floats, one fraction per hidden layer.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sums = None
    total = 0
    for x, _ in loader:
        x = x.to(DEVICE)
        _, activations = model(x, return_activations=True)
        if sums is None:
            sums = [act.abs().sum(dim=0) for act in activations]
        else:
            for i, act in enumerate(activations):
                sums[i] += act.abs().sum(dim=0)
        total += x.size(0)
    model.train()

    eps = 1e-9
    fractions = []
    for s in sums:
        mean_per_neuron = s / total
        layer_mean = mean_per_neuron.mean() + eps
        score = mean_per_neuron / layer_mean
        fractions.append((score <= tau).float().mean().item())
    return fractions


# --------------------------------------------------------------------------
# Experiment runners
# --------------------------------------------------------------------------

def run_joint_experiment(train_data, test_data, hidden_dims, lr, epochs, batch_size, seed,
                          tau=0.025):
    """Train one network on all tasks shuffled together. Reports, per task,
    the train/test accuracy and the dormant fraction before vs. after the
    whole joint training run."""
    set_seed(seed)
    model = MLP(hidden_dims=hidden_dims).to(DEVICE)

    dormant_start = [
        compute_dormant_fractions(model, train_data.get_task(t), batch_size, tau)
        for t in tqdm(range(train_data.num_tasks), desc="joint: dormancy@start", unit="task")
    ]

    optimizer = make_optimizer(model, lr)
    joint_train = ConcatDataset(train_data.tasks)
    for epoch in tqdm(range(epochs), desc="joint: epochs", unit="epoch"):
        train_one_epoch(model, joint_train, optimizer, batch_size, desc=f"joint epoch {epoch}")

    train_acc, test_acc, dormant_end = [], [], []
    for t in tqdm(range(train_data.num_tasks), desc="joint: per-task eval", unit="task"):
        train_task = train_data.get_task(t)
        test_task = test_data.get_task(t)
        train_acc.append(evaluate(model, train_task, batch_size))
        test_acc.append(evaluate(model, test_task, batch_size))
        dormant_end.append(compute_dormant_fractions(model, train_task, batch_size, tau))
        tqdm.write(f"[joint] task {t}: train_acc={train_acc[-1]:.4f} test_acc={test_acc[-1]:.4f}")

    metrics = {"train_acc": train_acc, "test_acc": test_acc,
               "dormant_start": dormant_start, "dormant_end": dormant_end}
    return metrics, model


def run_sequential_experiment(train_data, test_data, hidden_dims, lr, batch_size, seed,
                               reset_each_task, tau=0.025, label="sequential"):
    """One network trained one task at a time. If `reset_each_task`, a fresh
    network/optimizer is created for every task (experiment 2a); otherwise
    the same network is carried over across tasks, never reset (2b)."""
    set_seed(seed)
    model, optimizer = None, None
    metrics = {"train_acc": [], "test_acc": [], "dormant_start": [], "dormant_end": []}

    for task_id in tqdm(range(train_data.num_tasks), desc=f"{label}: tasks", unit="task"):
        if reset_each_task or model is None:
            model = MLP(hidden_dims=hidden_dims).to(DEVICE)
            optimizer = make_optimizer(model, lr)

        train_task = train_data.get_task(task_id)
        test_task = test_data.get_task(task_id)

        metrics["dormant_start"].append(compute_dormant_fractions(model, train_task, batch_size, tau))

        for epoch in range(train_data.epochs_per_task):
            train_one_epoch(model, train_task, optimizer, batch_size,
                             desc=f"{label} task {task_id} epoch {epoch}")

        metrics["train_acc"].append(evaluate(model, train_task, batch_size))
        metrics["test_acc"].append(evaluate(model, test_task, batch_size))
        metrics["dormant_end"].append(compute_dormant_fractions(model, train_task, batch_size, tau))
        tqdm.write(f"[{label}] task {task_id}: train_acc={metrics['train_acc'][-1]:.4f} "
                   f"test_acc={metrics['test_acc'][-1]:.4f}")

    return metrics, model


def run_frozen_representation_experiment(joint_model, train_data, test_data, hidden_dims, lr,
                                          batch_size, seed, tau=0.025, label="frozen_repr"):
    """Freeze every hidden layer from the jointly-trained model (a strict
    linear probe) and retrain only a fresh output layer, one task at a
    time."""
    set_seed(seed)
    model = copy.deepcopy(joint_model).to(DEVICE)
    model.freeze_representation(freeze=True)

    metrics = {"train_acc": [], "test_acc": [], "dormant_start": [], "dormant_end": []}

    for task_id in tqdm(range(train_data.num_tasks), desc=f"{label}: tasks", unit="task"):
        model.output_layer = nn.Linear(
            model.output_layer.in_features, model.output_layer.out_features
        ).to(DEVICE)
        optimizer = make_optimizer(model, lr)

        train_task = train_data.get_task(task_id)
        test_task = test_data.get_task(task_id)

        metrics["dormant_start"].append(compute_dormant_fractions(model, train_task, batch_size, tau))

        for epoch in range(train_data.epochs_per_task):
            train_one_epoch(model, train_task, optimizer, batch_size,
                             desc=f"{label} task {task_id} epoch {epoch}")

        metrics["train_acc"].append(evaluate(model, train_task, batch_size))
        metrics["test_acc"].append(evaluate(model, test_task, batch_size))
        metrics["dormant_end"].append(compute_dormant_fractions(model, train_task, batch_size, tau))
        tqdm.write(f"[{label}] task {task_id}: train_acc={metrics['train_acc'][-1]:.4f} "
                   f"test_acc={metrics['test_acc'][-1]:.4f}")

    return metrics, model


# --------------------------------------------------------------------------
# Output directory / checkpoint helpers
# --------------------------------------------------------------------------

def exp_dir(name):
    """outputs/<name>/ with a plots/ subfolder, created if missing."""
    path = os.path.join(OUTPUTS_DIR, name)
    os.makedirs(os.path.join(path, "plots"), exist_ok=True)
    return path


def save_metrics(metrics, path):
    torch.save(metrics, path)


def load_metrics(path):
    return torch.load(path)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_train_test_accuracy(results, out_path, title):
    """results: {label: {"train": [acc per task], "test": [acc per task]}}"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, series in results.items():
        axes[0].plot(range(len(series["train"])), series["train"], label=label)
        axes[1].plot(range(len(series["test"])), series["test"], label=label)
    axes[0].set_title("Train accuracy per task")
    axes[1].set_title("Test accuracy per task")
    for ax in axes:
        ax.set_xlabel("Task")
        ax.set_ylabel("Accuracy")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def plot_dormancy(results, num_layers, out_path, title):
    """results: {label: [[frac per layer] for each task]}"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 4 * num_layers), squeeze=False)
    axes = axes[:, 0]
    for label, history in results.items():
        history = np.array(history)  # (num_tasks, num_layers)
        for layer in range(num_layers):
            axes[layer].plot(range(len(history)), history[:, layer], label=label)
    for layer in range(num_layers):
        axes[layer].set_title(f"Hidden layer {layer}")
        axes[layer].set_xlabel("Task")
        axes[layer].set_ylabel("Dormant fraction")
        axes[layer].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")
