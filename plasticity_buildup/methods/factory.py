from .backprop import BackpropMethod


def make_training_method(method_type, model, optimizer, config):
    if method_type != "backprop":
        raise ValueError(f"Unknown training method type: {method_type!r}")
    return BackpropMethod(model=model, optimizer=optimizer, config=config)
