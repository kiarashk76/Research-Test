from .activation import activation_entropy, dormant_fraction
from .curvature import hessian_spectrum, ntk_change, ntk_matrix
from .gradients import fisher_information, gradient_norm
from .representation import effective_rank, feature_reuse, weight_movement


def collect_diagnostics(model, x, y, loss_fn, metrics, reference_parameters=None, reference_features=None, reference_ntk=None):
    """Compute only selected diagnostics; curvature remains opt-in."""
    available = {
        "dormant_fraction": lambda: dormant_fraction(model, x),
        "gradient_norm": lambda: gradient_norm(model, x, y, loss_fn),
        "effective_rank": lambda: effective_rank(model, x),
        "activation_entropy": lambda: activation_entropy(model, x),
        "fisher_information": lambda: fisher_information(model, x, y, loss_fn),
        "feature_reuse": lambda: feature_reuse(model, x, reference_features),
        "weight_movement": lambda: weight_movement(model, reference_parameters),
        "ntk_change": lambda: ntk_change(model, x, reference_ntk),
        "hessian_spectrum": lambda: hessian_spectrum(model, x, y, loss_fn),
    }
    selected = {}
    if "ntk_change" in metrics:
        if reference_ntk is None:
            raise ValueError("ntk_change requires a reference NTK in the collector")
    for metric_name, metric_config in metrics.items():
        if metric_config.get("type") != "diagnostic":
            continue
        if metric_name not in available:
            raise ValueError(f"No diagnostic implementation for {metric_name!r}.")
        selected[metric_name] = available[metric_name]()
    return selected
