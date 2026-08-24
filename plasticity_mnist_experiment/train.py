"""Continual-learning / loss-of-plasticity experiments on Permuted MNIST.

Experiments (see README below / module docstring):
  1. joint       - all tasks shuffled together at once (upper-bound sanity check)
  2. reset       - fresh network per task (no interference, no plasticity loss)
  3. no_reset    - one network trained sequentially, never reset (plasticity loss)
  4. redo        - one network trained sequentially, ReDo dormant-neuron resets
                   (allowed to reset any hidden layer)
  5. frozen_repr - representation frozen from the "joint" model, only the last
                   layer is retrained per task (linear probe per task)
  6. redo_first_layer - same as 4, but ReDo is only allowed to reset dormant
                   neurons in the first hidden layer

Each experiment reports "current task accuracy" per epoch, i.e. accuracy on
the test split of whichever task is currently being trained.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from dataset import PermutedMNIST
from network import MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# Experiments
# --------------------------------------------------------------------------

def run_joint(train_data, test_data, hidden_dims, lr, epochs, batch_size, seed):
    """Experiment 1: shuffle all tasks together, train one network on the
    combined data. Accuracy is measured on the combined test set."""
    set_seed(seed)
    model = MLP(hidden_dims=hidden_dims).to(DEVICE)
    optimizer = make_optimizer(model, lr)

    joint_train = ConcatDataset(train_data.tasks)
    joint_test = ConcatDataset(test_data.tasks)

    history = []
    for epoch in tqdm(range(epochs), desc="joint: epochs", unit="epoch"):
        train_one_epoch(model, joint_train, optimizer, batch_size,
                         desc=f"joint epoch {epoch}")
        acc = evaluate(model, joint_test, batch_size)
        history.append(acc)
        tqdm.write(f"[joint] epoch {epoch}: acc={acc:.4f}")
    return history, model


def run_reset_per_task(train_data, test_data, hidden_dims, lr, batch_size, seed):
    """Experiment 2: fresh network at the start of every task."""
    set_seed(seed)
    history = []
    for task_id in tqdm(range(train_data.num_tasks), desc="reset: tasks", unit="task"):
        model = MLP(hidden_dims=hidden_dims).to(DEVICE)
        optimizer = make_optimizer(model, lr)
        train_task = train_data.get_task(task_id)
        test_task = test_data.get_task(task_id)
        for epoch in range(train_data.epochs_per_task):
            train_one_epoch(model, train_task, optimizer, batch_size,
                             desc=f"reset task {task_id} epoch {epoch}")
            acc = evaluate(model, test_task, batch_size)
            history.append(acc)
            tqdm.write(f"[reset] task {task_id} epoch {epoch}: acc={acc:.4f}")
    return history


def run_no_reset(train_data, test_data, hidden_dims, lr, batch_size, seed):
    """Experiment 3: one network trained sequentially across tasks, never
    reset -> expected to show loss of plasticity on later tasks."""
    set_seed(seed)
    model = MLP(hidden_dims=hidden_dims).to(DEVICE)
    optimizer = make_optimizer(model, lr)

    history = []
    for task_id in tqdm(range(train_data.num_tasks), desc="no_reset: tasks", unit="task"):
        train_task = train_data.get_task(task_id)
        test_task = test_data.get_task(task_id)
        for epoch in range(train_data.epochs_per_task):
            train_one_epoch(model, train_task, optimizer, batch_size,
                             desc=f"no_reset task {task_id} epoch {epoch}")
            acc = evaluate(model, test_task, batch_size)
            history.append(acc)
            tqdm.write(f"[no_reset] task {task_id} epoch {epoch}: acc={acc:.4f}")
    return history, model


def run_redo(train_data, test_data, hidden_dims, lr, batch_size, seed, tau=0.025,
             layers=None, label="redo"):
    """Experiment 4 (and 6): one network trained sequentially, with ReDo
    resetting dormant neurons at the start of every epoch.

    `layers`: hidden-layer indices ReDo is allowed to touch (None = every
    hidden layer, i.e. experiment 4; [0] restricts it to the first hidden
    layer only, i.e. experiment 6).
    """
    set_seed(seed)
    model = MLP(hidden_dims=hidden_dims).to(DEVICE)
    optimizer = make_optimizer(model, lr)

    history = []
    for task_id in tqdm(range(train_data.num_tasks), desc=f"{label}: tasks", unit="task"):
        train_task = train_data.get_task(task_id)
        test_task = test_data.get_task(task_id)
        for epoch in range(train_data.epochs_per_task):
            num_reset = redo_pass(model, train_task, tau=tau, batch_size=batch_size,
                                   layers=layers)
            train_one_epoch(model, train_task, optimizer, batch_size,
                             desc=f"{label} task {task_id} epoch {epoch}")
            acc = evaluate(model, test_task, batch_size)
            history.append(acc)
            tqdm.write(f"[{label}] task {task_id} epoch {epoch}: acc={acc:.4f} (reset {num_reset} neurons)")
    return history, model


def run_frozen_representation(joint_model, train_data, test_data, hidden_dims, lr,
                               batch_size, seed):
    """Experiment 5: take the representation (all hidden layers) from the
    jointly-trained model, freeze it, and retrain only the output layer one
    task at a time."""
    set_seed(seed)
    model = copy.deepcopy(joint_model).to(DEVICE)
    model.freeze_representation(freeze=True)

    history = []
    for task_id in tqdm(range(train_data.num_tasks), desc="frozen_repr: tasks", unit="task"):
        # fresh readout layer per task, representation stays frozen/shared
        model.output_layer = nn.Linear(
            model.output_layer.in_features, model.output_layer.out_features
        ).to(DEVICE)
        optimizer = make_optimizer(model, lr)

        train_task = train_data.get_task(task_id)
        test_task = test_data.get_task(task_id)
        for epoch in range(train_data.epochs_per_task):
            train_one_epoch(model, train_task, optimizer, batch_size,
                             desc=f"frozen_repr task {task_id} epoch {epoch}")
            acc = evaluate(model, test_task, batch_size)
            history.append(acc)
            tqdm.write(f"[frozen_repr] task {task_id} epoch {epoch}: acc={acc:.4f}")
    return history


# --------------------------------------------------------------------------
# Plotting / entry point
# --------------------------------------------------------------------------

def plot_results(results, epochs_per_task, num_tasks, out_path="results.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for label, history in results.items():
        plt.plot(range(len(history)), history, label=label)
    for t in range(1, num_tasks):
        plt.axvline(t * epochs_per_task, color="gray", linestyle="--", linewidth=0.5)
    plt.xlabel("Epoch (global)")
    plt.ylabel("Current-task test accuracy")
    plt.title("Permuted MNIST: plasticity experiments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


def main():
    num_tasks = 800
    epochs_per_task = 1
    hidden_dims = (2000, 2000, 2000)
    lr = 1e-3
    batch_size = 1
    seed = 0

    train_data = PermutedMNIST(num_tasks=num_tasks, epochs_per_task=epochs_per_task,
                                seed=seed, train=True)
    test_data = PermutedMNIST(num_tasks=num_tasks, epochs_per_task=epochs_per_task,
                               seed=seed, train=False)

    results = {}

    print("\n=== [1/5] joint ===")
    joint_history, joint_model = run_joint(
        train_data, test_data, hidden_dims, lr,
        epochs=epochs_per_task * 5, batch_size=batch_size, seed=seed)
    results["joint"] = joint_history
    torch.save(joint_model.state_dict(), f"joint_model_{num_tasks}-tasks.pt")
    plot_results(results, epochs_per_task, num_tasks, "joint_results.png")
    print("Saved joint_model to joint_model.pt")
    exit(0)

    print("\n=== [2/5] reset_per_task ===")
    results["reset_per_task"] = run_reset_per_task(
        train_data, test_data, hidden_dims, lr, batch_size, seed)

    print("\n=== [3/5] no_reset ===")
    no_reset_history, _ = run_no_reset(
        train_data, test_data, hidden_dims, lr, batch_size, seed)
    results["no_reset"] = no_reset_history

    print("\n=== [4/6] redo (all layers) ===")
    redo_history, _ = run_redo(
        train_data, test_data, hidden_dims, lr, batch_size, seed,
        layers=None, label="redo")
    results["redo"] = redo_history

    print("\n=== [5/6] frozen_representation ===")
    results["frozen_representation"] = run_frozen_representation(
        joint_model, train_data, test_data, hidden_dims, lr, batch_size, seed)

    print("\n=== [6/6] redo_first_layer ===")
    redo_first_layer_history, _ = run_redo(
        train_data, test_data, hidden_dims, lr, batch_size, seed,
        layers=[0], label="redo_first_layer")
    results["redo_first_layer"] = redo_first_layer_history

    print("\nAll experiments finished. Plotting...")
    plot_results(results, epochs_per_task, num_tasks)


if __name__ == "__main__":
    main()
