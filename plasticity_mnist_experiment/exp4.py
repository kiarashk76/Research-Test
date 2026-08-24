"""Experiment 4 (stub / WIP): sequential training with ReDo dormant-neuron
resets at the start of every epoch. Kept from the old train.py so the
capability isn't lost, but not yet wired into the train/test-accuracy and
dormant-fraction tracking scheme used by exp1.py-exp3.py.
"""

import os

import torch
from tqdm.auto import tqdm

from dataset import PermutedMNIST
from network import MLP
from utils import DEVICE, evaluate, exp_dir, make_optimizer, redo_pass, set_seed, train_one_epoch

NUM_TASKS = 500
EPOCHS_PER_TASK = 1
HIDDEN_DIMS = (256, 256, 256)
LR = 1e-3
BATCH_SIZE = 128
SEED = 0
TAU = 0.025

EXP_NAME = f"exp4_redo_{NUM_TASKS}tasks"


def run_redo(train_data, test_data, hidden_dims, lr, batch_size, seed, tau=0.025,
             layers=None, label="redo"):
    """`layers`: hidden-layer indices ReDo is allowed to touch (None = every
    hidden layer)."""
    set_seed(seed)
    model = MLP(hidden_dims=hidden_dims).to(DEVICE)
    optimizer = make_optimizer(model, lr)

    history = []
    for task_id in tqdm(range(train_data.num_tasks), desc=f"{label}: tasks", unit="task"):
        train_task = train_data.get_task(task_id)
        test_task = test_data.get_task(task_id)
        for epoch in range(train_data.epochs_per_task):
            num_reset = redo_pass(model, train_task, tau=tau, batch_size=batch_size, layers=layers)
            train_one_epoch(model, train_task, optimizer, batch_size,
                             desc=f"{label} task {task_id} epoch {epoch}")
            acc = evaluate(model, test_task, batch_size)
            history.append(acc)
            tqdm.write(f"[{label}] task {task_id} epoch {epoch}: acc={acc:.4f} (reset {num_reset} neurons)")
    return history, model


def main():
    out_dir = exp_dir(EXP_NAME)

    train_data = PermutedMNIST(num_tasks=NUM_TASKS, epochs_per_task=EPOCHS_PER_TASK,
                                seed=SEED, train=True)
    test_data = PermutedMNIST(num_tasks=NUM_TASKS, epochs_per_task=EPOCHS_PER_TASK,
                               seed=SEED, train=False)

    history, model = run_redo(train_data, test_data, HIDDEN_DIMS, LR, BATCH_SIZE, SEED,
                               tau=TAU, layers=None, label="redo")
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    torch.save(history, os.path.join(out_dir, "history.pt"))
    print(f"Saved redo run to {out_dir}")


if __name__ == "__main__":
    main()
