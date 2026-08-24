"""Experiment 1: train one network jointly on all tasks (shuffled
together), then report per-task train/test accuracy and dormant-neuron
fractions before vs. after the joint training run.

Model + metrics are cached under outputs/ so re-running this script
without changing the config re-plots from disk instead of retraining.
"""

import os

import torch

from dataset import PermutedMNIST
from network import MLP
from utils import (DEVICE, exp_dir, load_metrics, plot_dormancy,
                    plot_train_test_accuracy, run_joint_experiment, save_metrics)

NUM_TASKS = 500
EPOCHS_PER_TASK = 1
JOINT_EPOCHS = EPOCHS_PER_TASK * 3
HIDDEN_DIMS = (256, 256, 256)
LR = 1e-3
BATCH_SIZE = 128
SEED = 0
TAU = 0.025

EXP_NAME = f"exp1_joint_{NUM_TASKS}tasks"


def main():
    out_dir = exp_dir(EXP_NAME)
    model_path = os.path.join(out_dir, "model.pt")
    metrics_path = os.path.join(out_dir, "metrics.pt")

    if os.path.exists(model_path) and os.path.exists(metrics_path):
        print(f"Found cached run at {out_dir}, skipping training.")
        metrics = load_metrics(metrics_path)
    else:
        train_data = PermutedMNIST(num_tasks=NUM_TASKS, epochs_per_task=EPOCHS_PER_TASK,
                                    seed=SEED, train=True)
        test_data = PermutedMNIST(num_tasks=NUM_TASKS, epochs_per_task=EPOCHS_PER_TASK,
                                   seed=SEED, train=False)
        metrics, model = run_joint_experiment(
            train_data, test_data, HIDDEN_DIMS, LR, JOINT_EPOCHS, BATCH_SIZE, SEED, tau=TAU)
        torch.save(model.state_dict(), model_path)
        save_metrics(metrics, metrics_path)
        print(f"Saved model to {model_path}, metrics to {metrics_path}")

    results = {"joint": {"train": metrics["train_acc"], "test": metrics["test_acc"]}}
    plot_train_test_accuracy(
        results, os.path.join(out_dir, "plots", "accuracy.png"),
        "exp1: joint model, accuracy per task")

    num_layers = len(HIDDEN_DIMS)
    plot_dormancy(
        {"joint": metrics["dormant_start"]}, num_layers,
        os.path.join(out_dir, "plots", "dormancy_start.png"),
        "exp1: dormant fraction before any training")
    plot_dormancy(
        {"joint": metrics["dormant_end"]}, num_layers,
        os.path.join(out_dir, "plots", "dormancy_end.png"),
        "exp1: dormant fraction after joint training")


if __name__ == "__main__":
    main()
