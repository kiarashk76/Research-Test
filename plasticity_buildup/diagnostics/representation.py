import torch


def _effective_rank(features):
    centered = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    total = singular_values.sum()
    if total <= 1e-12:
        return torch.zeros((), device=features.device)
    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    return (-(probabilities * probabilities.log()).sum()).exp()


def effective_rank(model, x):
    with torch.no_grad():
        per_layer = {}
        normalized = []
        for index, hidden in enumerate(model.hidden_features(x)):
            rank = _effective_rank(hidden)
            per_layer[f"hidden_{index}"] = rank
            normalized.append(rank / max(min(hidden.shape[0], hidden.shape[1]), 1))
        return {"global": torch.stack(normalized).mean(), "per_layer": per_layer}


def _cka(old_features, new_features):
    old = old_features - old_features.mean(dim=0, keepdim=True)
    new = new_features - new_features.mean(dim=0, keepdim=True)
    cross = old.T @ new
    old_covariance = old.T @ old
    new_covariance = new.T @ new
    numerator = cross.square().sum()
    denominator = old_covariance.square().sum().sqrt() * new_covariance.square().sum().sqrt()
    return numerator / denominator.clamp_min(1e-12)


def feature_reuse(model, x, reference_features):
    with torch.no_grad():
        current = model.extract_features(x)
        if len(current) != len(reference_features):
            raise ValueError("Reference and current feature lists must have the same number of layers.")
        per_layer = {f"hidden_{i}": _cka(old, new) for i, (old, new) in enumerate(zip(reference_features, current))}
        return {"global": torch.stack(list(per_layer.values())).mean(), "per_layer": per_layer}


def weight_movement(model, reference_parameters, relative=True):
    device = next(model.parameters()).device
    global_change = torch.zeros((), device=device)
    global_reference = torch.zeros((), device=device)
    layer_change = {}
    layer_reference = {}
    for name, parameter in model.named_parameters():
        reference = reference_parameters[name].to(parameter.device)
        change = (parameter - reference).square().sum()
        reference_norm = reference.square().sum()
        global_change += change
        global_reference += reference_norm
        layer_name = f"hidden_{name.split('.')[1]}" if name.startswith("hidden_layers.") else "output"
        layer_change[layer_name] = layer_change.get(layer_name, torch.zeros_like(change)) + change
        layer_reference[layer_name] = layer_reference.get(layer_name, torch.zeros_like(reference_norm)) + reference_norm
    per_layer = {name: value.sqrt() for name, value in layer_change.items()}
    if relative:
        per_layer = {name: value / layer_reference[name].sqrt().clamp_min(1e-12) for name, value in per_layer.items()}
    global_value = global_change.sqrt()
    if relative:
        global_value = global_value / global_reference.sqrt().clamp_min(1e-12)
    return {"global": global_value, "per_layer": per_layer}
