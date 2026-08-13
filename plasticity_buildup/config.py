SEED = 100
DATASET = "linear"
INPUT_DIM = 10
OUTPUT_DIM = 1
HIDDEN_DIMS = [256]
NUM_SAMPLES = 512
NUM_RUNS = 10
NUM_TASKS = 200
NUM_EPOCHS_PER_TASK = 50
LEARNING_RATE = 0.01
RESET_X = False
OPTIMIZER = "adam"
MOMENTUM = 0.9
LOSS_FUNCTION = "mse"

METRICS = {
    "first_loss": {"label": "Initial Task Loss", "type": "loss"},
    "average_loss": {"label": "Average Task Loss", "type": "loss"},
    "final_loss": {"label": "Final Task Loss", "type": "loss"},
    "dormant_fraction": {"label": "Dormant Fraction", "type": "diagnostic"},
    "gradient_norm": {"label": "Gradient Norm", "type": "diagnostic"},
    "effective_rank": {"label": "Effective Rank", "type": "diagnostic"},
    "activation_entropy": {"label": "Activation Entropy", "type": "diagnostic"},
    "fisher_information": {"label": "Mean Fisher Information", "type": "diagnostic"},
    "feature_reuse": {"label": "Feature Reuse CKA", "type": "diagnostic"},
    "weight_movement": {"label": "Within-Task Weight Movement", "type": "diagnostic"},
}

METHODS = {
    "backprop": {
        "label": "Backprop",
        "method_type": "backprop",
        "reset_model_each_task": False,
        "reset_optimizer_each_task": False,
    },
    "reset_optimizer": {
        "label": "Continual model + reset optimizer",
        "method_type": "backprop",
        "reset_model_each_task": False,
        "reset_optimizer_each_task": True,
    },
    "fresh": {
        "label": "Fresh Network",
        "method_type": "backprop",
        "reset_model_each_task": True,
        "reset_optimizer_each_task": True,
    },
    "random_reset": {
        "label": "Random Neuron Reset",
        "method_type": "random_reset",
        "reset_model_each_task": False,
        "reset_optimizer_each_task": False,
        "reset_frequency": 10,
        "reset_fraction": 0.01,
    },
    "low_gradient_reset": {
        "label": "Low-Gradient Reset",
        "method_type": "low_gradient_reset",
        "reset_model_each_task": False,
        "reset_optimizer_each_task": False,
        "reset_frequency": 10,
        "reset_fraction": 0.01,
    },
    "redo": {
        "label": "ReDo",
        "method_type": "redo",
        "reset_model_each_task": False,
        "reset_optimizer_each_task": False,
        "reset_frequency": 10,
        "dormancy_threshold": 0.01,
    },
    "continual_backprop": {
        "label": "Continual Backprop",
        "method_type": "continual_backprop",
        "reset_model_each_task": False,
        "reset_optimizer_each_task": False,
        "replacement_rate": 0.001,
        "maturity_threshold": 100,
        "utility_decay": 0.99,
    },
    "shrink_and_perturb": {
        "label": "Shrink and Perturb",
        "method_type": "shrink_and_perturb",
        "reset_model_each_task": False,
        "reset_optimizer_each_task": False,
        "shrink_factor": 0.9,
        "perturb_scale": 0.01,
    },
    "l2": {
        "label": "L2",
        "method_type": "l2",
        "reset_model_each_task": False,
        "reset_optimizer_each_task": False,
        "coefficient": 1e-4,
    },
    "l2_init": {
        "label": "L2-Init",
        "method_type": "l2_init",
        "reset_model_each_task": False,
        "reset_optimizer_each_task": False,
        "coefficient": 1e-4,
    },
}


def experiment_config():
    """Return the serializable experiment settings used by main.py."""
    return {
        "seed": SEED,
        "dataset": DATASET,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "hidden_dims": list(HIDDEN_DIMS) if isinstance(HIDDEN_DIMS, list) else HIDDEN_DIMS,
        "num_samples": NUM_SAMPLES,
        "num_runs": NUM_RUNS,
        "num_tasks": NUM_TASKS,
        "num_epochs_per_task": NUM_EPOCHS_PER_TASK,
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER,
        "momentum": MOMENTUM,
        "loss_function": LOSS_FUNCTION,
        "reset_x": RESET_X,
    }
