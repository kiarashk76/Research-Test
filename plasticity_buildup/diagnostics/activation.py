import torch


def dormant_fraction(model, x, threshold=1e-3):
    with torch.no_grad():
        features = model.hidden_features(x)
        per_layer = {}
        dormant_count = 0
        neuron_count = 0
        for index, hidden in enumerate(features):
            activity = hidden.abs().mean(dim=0)
            normalized = activity / activity.mean().clamp_min(1e-12)
            dormant = normalized <= threshold
            per_layer[f"hidden_{index}"] = dormant.float().mean()
            dormant_count += dormant.sum()
            neuron_count += dormant.numel()
        return {"global": dormant_count.float() / neuron_count, "per_layer": per_layer}


def _entropy_for_layer(hidden, bins):
    entropies = []
    for neuron in range(hidden.shape[1]):
        values = hidden[:, neuron]
        minimum, maximum = values.min(), values.max()
        if torch.isclose(minimum, maximum):
            entropy = torch.zeros((), device=hidden.device)
        else:
            histogram = torch.histc(values.float(), bins=bins, min=minimum.item(), max=maximum.item())
            probabilities = histogram / histogram.sum().clamp_min(1)
            probabilities = probabilities[probabilities > 0]
            entropy = -(probabilities * probabilities.log()).sum()
        entropies.append(entropy)
    return torch.stack(entropies)


def activation_entropy(model, x, bins=30):
    with torch.no_grad():
        entropies = [_entropy_for_layer(hidden, bins) for hidden in model.hidden_features(x)]
        per_layer = {f"hidden_{i}": values.mean() for i, values in enumerate(entropies)}
        return {
            "global": torch.cat(entropies).mean(),
            "per_layer": per_layer,
            "details": {f"hidden_{i}": values for i, values in enumerate(entropies)},
        }
