"""Central configuration for metric axis scales."""


# Edit this mapping when changing the y-axis scale for a metric. Supported
# values are Matplotlib scale names, for example: "linear", "log", or
# "symlog".
METRIC_Y_SCALES = {
    "final_loss": "log",
    "gradient_norm": "linear",
    "effective_rank": "linear",
    "dormant_fraction": "linear",
}

# Optional keyword arguments for individual Matplotlib scales.
METRIC_Y_SCALE_OPTIONS = {
    # "dormant_fraction": {"linthresh": 1e-6},
}


def apply_metric_y_scale(axis, metric_name, enabled=True):
    """Apply the configured y-axis scale for a metric to a Matplotlib axis."""
    if not enabled:
        axis.set_yscale("linear")
        return
    scale = METRIC_Y_SCALES.get(metric_name, "linear")
    options = METRIC_Y_SCALE_OPTIONS.get(metric_name, {})
    axis.set_yscale(scale, **options)
