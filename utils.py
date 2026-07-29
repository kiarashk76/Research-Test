from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import torch


RESULT_FIELDNAMES = [
    "seed",
    "task",
    "continual_initial_mse",
    "continual_final_mse",
    "fresh_initial_mse",
    "fresh_final_mse",
    "final_mse_gap",
    "final_mse_ratio",
]
INT_FIELDS = {"seed", "task"}
METRIC_FIELDS = set(RESULT_FIELDNAMES) - INT_FIELDS


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*values: int) -> int:
    seed = 0x345678
    for value in values:
        seed = ((seed ^ value) * 1000003) & 0xFFFFFFFF
    return seed


def write_csv(path: Path, rows: Iterable[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_results_csv(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float | int] = {}
            for key, value in row.items():
                if key in INT_FIELDS:
                    parsed[key] = int(value)
                elif key in METRIC_FIELDS:
                    parsed[key] = float(value)
                else:
                    raise ValueError(f"Unexpected column {key!r} in {path}")
            rows.append(parsed)
    return rows


def summarize_by_task(rows: list[dict[str, float | int]], metric: str) -> tuple[list[int], list[float], list[float]]:
    tasks = sorted({int(row["task"]) for row in rows})
    means: list[float] = []
    sems: list[float] = []

    for task in tasks:
        values = [float(row[metric]) for row in rows if int(row["task"]) == task]
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            sem = math.sqrt(variance / len(values))
        else:
            sem = 0.0
        means.append(mean)
        sems.append(sem)

    return tasks, means, sems


def mean_final_mse(rows: list[dict[str, float | int]], task: int, metric: str) -> float:
    values = [float(row[metric]) for row in rows if int(row["task"]) == task]
    return sum(values) / len(values)
