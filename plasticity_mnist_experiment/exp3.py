"""Experiment 3: freeze the joint model's representation (every hidden
layer) and retrain only a fresh linear output layer, one task at a time
(a linear probe per task). Requires exp1.py to have already been run.

Model + metrics are cached under outputs/ so re-running this script
without changing the config re-plots from disk instead of retraining.
"""

import os

import torch

from dataset import PermutedMNIST
from network import MLP
from utils import (DEVICE, exp_dir, load_metrics, plot_dormancy,
                    plot_train_test_accuracy, run_frozen_representation_experiment,
                    save_metrics)

NUM_TASKS = 500
EPOCHS_PER_TASK = 1
HIDDEN_DIMS = (256, 256, 256)
LR = 1e-3
BATCH_SIZE = 128
SEED = 0
TAU = 0.025

EXP_NAME = f"exp3_frozen_repr_{NUM_TASKS}tasks"
JOINT_MODEL_PATH = os.path.join("outputs", f"exp1_joint_{NUM_TASKS}tasks", "model.pt")


def main():
    if not os.path.exists(JOINT_MODEL_PATH):
        raise FileNotFoundError(
            f"{JOINT_MODEL_PATH} not found. Run exp1.py first to train the joint model.")

    out_dir = exp_dir(EXP_NAME)
    model_path = os.path.join(out_dir, "model.pt")
    metrics_path = os.path.join(out_dir, "metrics.pt")

    if os.path.exists(model_path) and os.path.exists(metrics_path):
        print(f"Found cached run at {out_dir}, skipping training.")
        metrics = load_metrics(metrics_path)
    else:
        joint_model = MLP(hidden_dims=HIDDEN_DIMS).to(DEVICE)
        joint_model.load_state_dict(torch.load(JOINT_MODEL_PATH, map_location=DEVICE))

        train_data = PermutedMNIST(num_tasks=NUM_TASKS, epochs_per_task=EPOCHS_PER_TASK,
                                    seed=SEED, train=True)
        test_data = PermutedMNIST(num_tasks=NUM_TASKS, epochs_per_task=EPOCHS_PER_TASK,
                                   seed=SEED, train=False)

        metrics, model = run_frozen_representation_experiment(
            joint_model, train_data, test_data, HIDDEN_DIMS, LR, BATCH_SIZE, SEED, tau=TAU)
        torch.save(model.state_dict(), model_path)
        save_metrics(metrics, metrics_path)
        print(f"Saved model to {model_path}, metrics to {metrics_path}")

    results = {"frozen_repr": {"train": metrics["train_acc"], "test": metrics["test_acc"]}}
    plot_train_test_accuracy(
        results, os.path.join(out_dir, "plots", "accuracy.png"),
        "exp3: frozen representation, linear probe per task")

    num_layers = len(HIDDEN_DIMS)
    plot_dormancy(
        {"frozen_repr": metrics["dormant_start"]}, num_layers,
        os.path.join(out_dir, "plots", "dormancy_start.png"),
        "exp3: dormant fraction at task start")
    plot_dormancy(
        {"frozen_repr": metrics["dormant_end"]}, num_layers,
        os.path.join(out_dir, "plots", "dormancy_end.png"),
        "exp3: dormant fraction at task end")


if __name__ == "__main__":
    main()
