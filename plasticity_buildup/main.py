"""Simple entry point for the three incremental experiment workflows."""

try:
    from . import config as default_config
    from .experiments.workflows import create_shared_experiment, plot_selected_methods, run_selected_methods
    from .utils.naming import make_experiment_directory
except ImportError:  # Supports `python main.py` from plasticity_buildup/.
    import config as default_config
    from experiments.workflows import create_shared_experiment, plot_selected_methods, run_selected_methods
    from utils.naming import make_experiment_directory


# Edit these settings for the intended workflow.
MODE = "plot"  # "generate", "run_methods", "plot", or "all"
EXPERIMENT_DIR = "outputs/nonlinear_xfixed/in10_h8-8_out1/samples512_tasks200_epochs50/adam_lr0.01"
METHODS_TO_RUN = [
    "backprop",
    # "reset_optimizer",
    "fresh",
    "random_reset",
    # "low_gradient_reset",
    "redo",
    "continual_backprop",
    # "shrink_and_perturb",
    # "l2",
    "l2_init",
]
METHODS_TO_PLOT = METHODS_TO_RUN
METRICS_TO_PLOT = [
    "final_loss",
    "dormant_fraction",
    "gradient_norm",
    "effective_rank",
]


def main(
    mode=MODE,
    experiment_dir=EXPERIMENT_DIR,
    methods_to_run=None,
    methods_to_plot=None,
    metrics_to_plot=METRICS_TO_PLOT,
):
    settings = default_config.experiment_config()
    methods_to_run = methods_to_run or METHODS_TO_RUN
    methods_to_plot = methods_to_plot or METHODS_TO_PLOT
    if experiment_dir is None:
        experiment_dir = make_experiment_directory(settings)

    if mode in ("generate", "all"):
        import os
        if mode == "generate" or not os.path.exists(os.path.join(experiment_dir, "metadata.json")):
            create_shared_experiment(settings, default_config.METRICS, experiment_dir)

    if mode in ("run_methods", "all"):
        run_selected_methods(
            experiment_dir,
            methods_to_run,
            default_config.METHODS,
            default_config.METRICS,
        )

    if mode in ("plot", "all"):
        plot_selected_methods(experiment_dir, methods_to_plot, metrics_to_plot)

    return experiment_dir


if __name__ == "__main__":
    main()
