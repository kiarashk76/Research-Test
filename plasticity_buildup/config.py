SEED = 100
DATASET = "linear"
INPUT_DIM = 10
OUTPUT_DIM = 1
HIDDEN_DIMS = [64, 64, 64]
NUM_SAMPLES = 512
NUM_RUNS = 3
NUM_TASKS = 200
NUM_EPOCHS_PER_TASK = 50
LEARNING_RATE = 0.01
RESET_X = False
OPTIMIZER = "adam"
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
    "continual": {
        "label": "Continual model + continual optimizer",
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
        "label": "Fresh model + fresh optimizer",
        "method_type": "backprop",
        "reset_model_each_task": True,
        "reset_optimizer_each_task": True,
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
        "loss_function": LOSS_FUNCTION,
        "reset_x": RESET_X,
    }
