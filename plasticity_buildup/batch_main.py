"""Run a list of experiment configurations sequentially.

Edit EXPERIMENTS below, then run:

    python batch_main.py

Each experiment uses the normal outputs/ directory layout. Existing compatible
experiments and method result files are reused.
"""

import os

try:
    from . import config as default_config
    from .experiments.workflows import (
        create_shared_experiment,
        load_shared_experiment,
        plot_selected_methods,
        run_selected_methods,
    )
    from .utils.naming import make_experiment_directory
except ImportError:  # Supports `python batch_main.py` from plasticity_buildup/.
    import config as default_config
    from experiments.workflows import (
        create_shared_experiment,
        load_shared_experiment,
        plot_selected_methods,
        run_selected_methods,
    )
    from utils.naming import make_experiment_directory


# Add or remove entries here. Each entry may override any base config setting.
EXPERIMENTS = [
    # {
    #     "overrides": {"hidden_dims": [64, 64], "dataset": "nonlinear"},
    #     # "methods": ["backprop", "continual_backprop"],
    # },
    {
        "overrides": {"hidden_dims": [48, 48, 48], "dataset": "nonlinear"},
    },
    {
        "overrides": {"hidden_dims": [40, 40, 40, 40], "dataset": "nonlinear"},
    },
]

METHODS_TO_RUN = [
    "backprop",
    "reset_optimizer",
    "fresh",
    "random_reset",
    "low_gradient_reset",
    "redo",
    "continual_backprop",
    "shrink_and_perturb",
    "l2",
    "l2_init",
]
PLOT_AFTER_RUN = True
METRICS_TO_PLOT = None


def _get_or_create_experiment(settings):
    experiment_dir = make_experiment_directory(settings)
    metadata_path = os.path.join(experiment_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        create_shared_experiment(settings, default_config.METRICS, experiment_dir)
        return experiment_dir

    metadata = load_shared_experiment(experiment_dir)
    if metadata["config"] != settings or metadata["metrics"] != default_config.METRICS:
        raise ValueError(
            "An experiment already exists at the automatically generated path, "
            "but its configuration does not match the requested settings: "
            f"{experiment_dir}"
        )
    return experiment_dir


def run_batch(experiments=None):
    experiments = EXPERIMENTS if experiments is None else experiments
    if not experiments:
        print("No experiments configured. Edit EXPERIMENTS in batch_main.py.")
        return []

    completed = []
    for index, experiment in enumerate(experiments, start=1):
        settings = default_config.experiment_config()
        settings.update(experiment.get("overrides", {}))
        methods_to_run = experiment.get("methods", METHODS_TO_RUN)
        print(f"\nExperiment {index}/{len(experiments)}: {settings}")
        experiment_dir = _get_or_create_experiment(settings)
        run_selected_methods(
            experiment_dir,
            methods_to_run,
            default_config.METHODS,
            default_config.METRICS,
        )
        if PLOT_AFTER_RUN:
            plot_selected_methods(experiment_dir, methods_to_run, METRICS_TO_PLOT)
        completed.append(experiment_dir)
        print(f"Completed: {experiment_dir}")
    return completed


if __name__ == "__main__":
    run_batch()
