"""Shared neuron-reset and optimizer-state helpers for the MLP methods."""

import torch


def reset_hidden_neuron(model, layer_index, neuron_index, zero_outgoing=True):
    """Reinitialize one hidden neuron using the model's Linear rule."""
    model.reset_hidden_neuron(layer_index, neuron_index, zero_outgoing=zero_outgoing)


def _clear_parameter_slice(optimizer, parameter, index, axis=0):
    state = optimizer.state.get(parameter)
    if not state:
        return
    for value in state.values():
        if not torch.is_tensor(value) or value.shape != parameter.shape:
            continue
        if axis == 0:
            value[index].zero_()
        else:
            value[:, index].zero_()


def clear_neuron_optimizer_state(optimizer, model, layer_index, neuron_index):
    """Clear Adam moments only for a neuron's incoming and outgoing slices."""
    layer = model.hidden_layers[layer_index]
    _clear_parameter_slice(optimizer, layer.weight, neuron_index, axis=0)
    if layer.bias is not None:
        _clear_parameter_slice(optimizer, layer.bias, neuron_index, axis=0)

    next_layer = model.get_next_layer(layer_index)
    if next_layer is not None:
        _clear_parameter_slice(optimizer, next_layer.weight, neuron_index, axis=1)


def reset_and_clear_neuron(model, optimizer, layer_index, neuron_index, zero_outgoing=True):
    reset_hidden_neuron(model, layer_index, neuron_index, zero_outgoing=zero_outgoing)
    clear_neuron_optimizer_state(optimizer, model, layer_index, neuron_index)


def all_hidden_neurons(model):
    return [(layer_index, neuron_index)
            for layer_index, layer in enumerate(model.hidden_layers)
            for neuron_index in range(layer.out_features)]


def number_to_reset(total, fraction):
    if fraction <= 0 or total == 0:
        return 0
    return min(total, max(1, int(total * fraction)))
