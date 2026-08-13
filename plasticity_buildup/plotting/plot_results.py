import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..experiments.workflows import plot_selected_methods
except ImportError:
    from experiments.workflows import plot_selected_methods


# Edit these lists when producing a focused comparison. None means all available.
EXPERIMENT_DIR = None
METHODS_TO_PLOT = [
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
METRICS_TO_PLOT = None


def recreate_plots(experiment_dir, methods_to_plot=None, metrics_to_plot=None):
    return plot_selected_methods(experiment_dir, methods_to_plot or METHODS_TO_PLOT, metrics_to_plot)


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else EXPERIMENT_DIR
    if directory is None:
        raise SystemExit("Set EXPERIMENT_DIR near the top of plot_results.py or pass an experiment directory.")
    recreate_plots(directory, METHODS_TO_PLOT, METRICS_TO_PLOT)
