import torch


def _hidden_layer_name(parameter_name):
    if parameter_name.startswith("hidden_layers."):
        return f"hidden_{parameter_name.split('.')[1]}"
    return "output"


def gradient_norm(model, x, y, loss_fn):
    loss = loss_fn(model(x), y)
    named_parameters = list(model.named_parameters())
    gradients = torch.autograd.grad(loss, tuple(parameter for _, parameter in named_parameters), allow_unused=True)
    global_squared = torch.zeros((), device=next(model.parameters()).device)
    layer_squared = {}
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        squared = gradient.square().sum()
        global_squared += squared
        layer_name = _hidden_layer_name(name)
        if layer_name is not None:
            layer_squared[layer_name] = layer_squared.get(layer_name, torch.zeros_like(squared)) + squared
    return {"global": global_squared.sqrt(), "per_layer": {name: value.sqrt() for name, value in layer_squared.items()}}


def fisher_information(model, x, y, loss_fn):
    loss = loss_fn(model(x), y)
    named_parameters = list(model.named_parameters())
    gradients = torch.autograd.grad(loss, tuple(parameter for _, parameter in named_parameters), allow_unused=True)
    values_by_layer = {}
    all_values = []
    diagonal = {}
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        values = gradient.detach().square()
        diagonal[name] = values
        all_values.append(values.reshape(-1))
        layer_name = _hidden_layer_name(name)
        if layer_name is not None:
            values_by_layer.setdefault(layer_name, []).append(values.reshape(-1))
    per_layer = {name: torch.cat(values).mean() for name, values in values_by_layer.items()}
    return {
        "global": torch.cat(all_values).mean(),
        "per_layer": per_layer,
        "details": {"diagonal": diagonal},
    }
