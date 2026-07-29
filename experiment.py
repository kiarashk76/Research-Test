from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from networks import ValueMLP
from utils import seed_torch, stable_seed, summarize_by_task, write_csv


@dataclass(frozen=True)
class ExperimentConfig:
    num_states: int = 100
    num_tasks: int = 100
    steps_per_task: int = 500
    hidden_size: int = 64
    learning_rate: float = 0.05
    optimizer: str = "sgd"
    activation: str = "relu"
    num_seeds: int = 5
    seed_offset: int = 0
    device: str = "cpu"
    output_dir: Path = Path("results/value_prediction_plasticity")
    reset_continual_optimizer_per_task: bool = False


def make_model(config: ExperimentConfig, seed: int) -> ValueMLP:
    seed_torch(seed)
    return ValueMLP(config.num_states, config.hidden_size, config.activation).to(config.device)


def make_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
) -> tuple[float, float]:
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        initial_mse = loss_fn(model(states), targets).item()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(states), targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_mse = loss_fn(model(states), targets).item()

    return initial_mse, final_mse


def run_seed(seed: int, config: ExperimentConfig) -> list[dict[str, float | int]]:
    generator = torch.Generator(device=config.device)
    generator.manual_seed(seed)

    states = torch.eye(config.num_states, device=config.device)
    targets_by_task = torch.randn(
        config.num_tasks,
        config.num_states,
        generator=generator,
        device=config.device,
    )

    continual_model = make_model(config, stable_seed(seed, 0, 0))
    continual_optimizer = make_optimizer(continual_model, config)

    rows: list[dict[str, float | int]] = []
    for task_index, targets in enumerate(targets_by_task, start=1):
        if config.reset_continual_optimizer_per_task and task_index > 1:
            continual_optimizer = make_optimizer(continual_model, config)

        continual_initial, continual_final = train_steps(
            continual_model,
            continual_optimizer,
            states,
            targets,
            config.steps_per_task,
        )

        fresh_model = make_model(config, stable_seed(seed, task_index, 1))
        fresh_optimizer = make_optimizer(fresh_model, config)
        fresh_initial, fresh_final = train_steps(
            fresh_model,
            fresh_optimizer,
            states,
            targets,
            config.steps_per_task,
        )

        rows.append(
            {
                "seed": seed,
                "task": task_index,
                "continual_initial_mse": continual_initial,
                "continual_final_mse": continual_final,
                "fresh_initial_mse": fresh_initial,
                "fresh_final_mse": fresh_final,
                "final_mse_gap": continual_final - fresh_final,
                "final_mse_ratio": continual_final / max(fresh_final, 1e-12),
            }
        )

    return rows


def plot_results(rows: list[dict[str, float | int]], config: ExperimentConfig, path: Path) -> None:
    tasks, continual_mean, continual_sem = summarize_by_task(rows, "continual_final_mse")
    _, fresh_mean, fresh_sem = summarize_by_task(rows, "fresh_final_mse")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(tasks, continual_mean, label="Continual network", linewidth=2)
    ax.fill_between(
        tasks,
        [mean - sem for mean, sem in zip(continual_mean, continual_sem)],
        [mean + sem for mean, sem in zip(continual_mean, continual_sem)],
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(tasks, fresh_mean, label="Fresh network control", linewidth=2)
    ax.fill_between(
        tasks,
        [mean - sem for mean, sem in zip(fresh_mean, fresh_sem)],
        [mean + sem for mean, sem in zip(fresh_mean, fresh_sem)],
        alpha=0.18,
        linewidth=0,
    )
    ax.set_xlabel("Task number")
    ax.set_ylabel(f"MSE after {config.steps_per_task} updates")
    ax.set_title("Plasticity test: learning new random value targets")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_config(config: ExperimentConfig) -> None:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    with (config.output_dir / "config.json").open("w") as handle:
        json.dump(data, handle, indent=2)


def run_experiment(config: ExperimentConfig) -> list[dict[str, float | int]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config)

    all_rows: list[dict[str, float | int]] = []
    seeds = range(config.seed_offset, config.seed_offset + config.num_seeds)
    for run_number, seed in enumerate(seeds, start=1):
        print(f"Running seed {seed} ({run_number}/{config.num_seeds})", flush=True)
        all_rows.extend(run_seed(seed, config))

    write_csv(config.output_dir / "plasticity_results.csv", all_rows)
    plot_results(all_rows, config, config.output_dir / "mse_after_500_updates.png")
    return all_rows
