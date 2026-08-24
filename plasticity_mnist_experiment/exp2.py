"""Experiment 2: sequential training across tasks, comparing a network
reset at the start of every task against one that's never reset. The
joint model's per-task train/test accuracy and dormancy (from exp1.py) are
overlaid for reference if that run has already been cached.

Models + metrics are cached under outputs/ so re-running this script
without changing the config re-plots from disk instead of retraining.
"""

import os

import torch

from dataset import PermutedMNIST
from utils import (exp_dir, load_metrics, plot_dormancy, plot_train_test_accuracy,
                    run_sequential_experiment, save_metrics)

NUM_TASKS = 500
EPOCHS_PER_TASK = 1
HIDDEN_DIMS = (256, 256, 256)
LR = 1e-3
BATCH_SIZE = 128
SEED = 0
TAU = 0.025

EXP_NAME = f"exp2_sequential_{NUM_TASKS}tasks"
JOINT_METRICS_PATH = os.path.join("outputs", f"exp1_joint_{NUM_TASKS}tasks", "metrics.pt")


def run_or_load(variant, reset_each_task, train_data, test_data, out_dir):
    model_path = os.path.join(out_dir, f"{variant}_model.pt")
    metrics_path = os.path.join(out_dir, f"{variant}_metrics.pt")

    if os.path.exists(metrics_path):
        print(f"Loading cached {variant} metrics from {metrics_path}")
        return load_metrics(metrics_path)

    metrics, model = run_sequential_experiment(
        train_data, test_data, HIDDEN_DIMS, LR, BATCH_SIZE, SEED,
        reset_each_task=reset_each_task, tau=TAU, label=variant)
    torch.save(model.state_dict(), model_path)
    save_metrics(metrics, metrics_path)
    print(f"Saved {variant} model to {model_path}, metrics to {metrics_path}")
    return metrics


def main():
    out_dir = exp_dir(EXP_NAME)

    train_data = PermutedMNIST(num_tasks=NUM_TASKS, epochs_per_task=EPOCHS_PER_TASK,
                                seed=SEED, train=True)
    test_data = PermutedMNIST(num_tasks=NUM_TASKS, epochs_per_task=EPOCHS_PER_TASK,
                               seed=SEED, train=False)

    no_reset_metrics = run_or_load("no_reset", False, train_data, test_data, out_dir)
    reset_metrics = run_or_load("reset_per_task", True, train_data, test_data, out_dir)

    acc_results = {
        "no_reset": {"train": no_reset_metrics["train_acc"], "test": no_reset_metrics["test_acc"]},
        "reset_per_task": {"train": reset_metrics["train_acc"], "test": reset_metrics["test_acc"]},
    }
    dormant_start_results = {
        "no_reset": no_reset_metrics["dormant_start"],
        "reset_per_task": reset_metrics["dormant_start"],
    }
    dormant_end_results = {
        "no_reset": no_reset_metrics["dormant_end"],
        "reset_per_task": reset_metrics["dormant_end"],
    }

    if os.path.exists(JOINT_METRICS_PATH):
        joint_metrics = load_metrics(JOINT_METRICS_PATH)
        acc_results["joint"] = {"train": joint_metrics["train_acc"], "test": joint_metrics["test_acc"]}
        dormant_start_results["joint"] = joint_metrics["dormant_start"]
        dormant_end_results["joint"] = joint_metrics["dormant_end"]
    else:
        print(f"No joint metrics found at {JOINT_METRICS_PATH}; plotting without the joint "
              "reference. Run exp1.py first to include it.")

    plot_train_test_accuracy(
        acc_results, os.path.join(out_dir, "plots", "accuracy.png"),
        "exp2: no_reset vs. reset_per_task vs. joint")

    num_layers = len(HIDDEN_DIMS)
    plot_dormancy(
        dormant_start_results, num_layers, os.path.join(out_dir, "plots", "dormancy_start.png"),
        "exp2: dormant fraction at task start")
    plot_dormancy(
        dormant_end_results, num_layers, os.path.join(out_dir, "plots", "dormancy_end.png"),
        "exp2: dormant fraction at task end")


if __name__ == "__main__":
    main()
