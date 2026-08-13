from .backprop import BackpropMethod
from .continual_backprop import ContinualBackpropMethod
from .l2_init import L2InitMethod
from .l2_regularization import L2RegularizationMethod
from .low_gradient_reset import LowGradientResetMethod
from .random_reset import RandomResetMethod
from .redo import ReDoMethod
from .shrink_and_perturb import ShrinkAndPerturbMethod


def make_training_method(method_type, model, optimizer, config, initial_state=None):
    if method_type == "backprop":
        method_class = BackpropMethod
    elif method_type == "random_reset":
        method_class = RandomResetMethod
    elif method_type == "low_gradient_reset":
        method_class = LowGradientResetMethod
    elif method_type == "redo":
        method_class = ReDoMethod
    elif method_type == "continual_backprop":
        method_class = ContinualBackpropMethod
    elif method_type == "shrink_and_perturb":
        method_class = ShrinkAndPerturbMethod
    elif method_type == "l2":
        method_class = L2RegularizationMethod
    elif method_type == "l2_init":
        method_class = L2InitMethod
    else:
        raise ValueError(f"Unknown training method type: {method_type!r}")
    return method_class(model=model, optimizer=optimizer, config=config, initial_state=initial_state)
