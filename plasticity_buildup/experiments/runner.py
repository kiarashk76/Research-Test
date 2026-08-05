import copy

import torch

try:
    from ..datasets.factory import make_dataset
    from ..diagnostics.collector import collect_diagnostics
    from ..methods.factory import make_training_method
    from ..models.mlp import Network
    from ..train import train_on_task
    from ..utils.seeds import set_seed
except ImportError:  # Supports running main.py directly from this directory.
    from datasets.factory import make_dataset
    from diagnostics.collector import collect_diagnostics
    from methods.factory import make_training_method
    from models.mlp import Network
    from train import train_on_task
    from utils.seeds import set_seed
from .results import convert_results_to_numpy, create_empty_results, stack_runs, store_task_results
from .state import MethodState


def make_model(config):
    return Network(config["input_dim"], config["hidden_dims"], config["output_dim"])


def make_optimizer(model, config):
    if config.get("optimizer", "adam").lower() != "adam":
        raise ValueError("Only the existing Adam optimizer is currently supported")
    return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])


def make_loss_function(config):
    if config.get("loss_function", "mse").lower() != "mse":
        raise ValueError("Only the existing MSE loss is currently supported")
    return torch.nn.MSELoss()


def generate_tasks(config, seed):
    set_seed(seed)
    dataset = make_dataset(config["dataset"], config)
    tasks = []
    for _ in range(config["num_tasks"]):
        x, y = dataset.generate_data(reset_x=config["reset_x"])
        tasks.append({"x": x.detach().clone(), "y": y.detach().clone()})
    return tasks


def _task_values(task):
    if isinstance(task, dict):
        return task["x"], task["y"]
    return task


def _make_method_states(methods, metrics, config, initial_state):
    states = {}
    for method_name, method_config in methods.items():
        model = make_model(config)
        model.load_state_dict(initial_state)
        optimizer = make_optimizer(model, config)
        training_method = make_training_method(method_config.get("method_type", "backprop"), model, optimizer, method_config)
        states[method_name] = MethodState(model, optimizer, training_method, create_empty_results(metrics))
    return states


def run_method_on_tasks(config, method_name, method_config, metrics, tasks, initial_state, run_seed):
    """Run exactly one method on one saved run without generating any data."""
    set_seed(run_seed)
    states = _make_method_states({method_name: method_config}, metrics, config, initial_state)
    state = states[method_name]
    loss_fn = make_loss_function(config)
    for task in tasks:
        x, y = _task_values(task)
        if method_config.get("reset_model_each_task", False):
            state.model.load_state_dict(initial_state)
        if method_config.get("reset_optimizer_each_task", False):
            state.optimizer = make_optimizer(state.model, config)
            state.training_method.optimizer = state.optimizer

        reference_parameters = state.model.save_reference_parameters()
        reference_features = state.model.extract_features(x)
        reference_ntk = None
        if "ntk_change" in metrics:
            try:
                from ..diagnostics.curvature import ntk_matrix
            except ImportError:
                from diagnostics.curvature import ntk_matrix
            reference_ntk = ntk_matrix(state.model, x).detach()

        losses = train_on_task(
            state.model,
            state.optimizer,
            state.training_method,
            x,
            y,
            loss_fn,
            config["num_epochs_per_task"],
        )
        diagnostics = collect_diagnostics(
            state.model,
            x,
            y,
            loss_fn,
            metrics,
            reference_parameters=reference_parameters,
            reference_features=reference_features,
            reference_ntk=reference_ntk,
        )
        store_task_results(state.results, losses, diagnostics, metrics)
    return convert_results_to_numpy(state.results)


def run_experiment(config, methods, metrics, task_runs=None, initial_state=None):
    """Run every configured method on shared tasks and initialization."""
    if initial_state is None:
        set_seed(config["seed"])
        initial_model = make_model(config)
        initial_state = copy.deepcopy(initial_model.state_dict())
    else:
        initial_state = copy.deepcopy(initial_state)

    if task_runs is None:
        task_runs = [generate_tasks(config, config["seed"] + run) for run in range(config["num_runs"])]
    elif task_runs and isinstance(task_runs[0], dict):
        task_runs = [task_runs]

    run_results = {method_name: [] for method_name in methods}
    loss_fn = make_loss_function(config)

    for run_index, tasks in enumerate(task_runs):
        set_seed(config["seed"] + run_index)
        method_states = _make_method_states(methods, metrics, config, initial_state)
        for task in tasks:
            x, y = _task_values(task)
            for method_name, state in method_states.items():
                method_config = methods[method_name]
                if method_config.get("reset_model_each_task", False):
                    state.model.load_state_dict(initial_state)
                if method_config.get("reset_optimizer_each_task", False):
                    state.optimizer = make_optimizer(state.model, config)
                    state.training_method.optimizer = state.optimizer

                reference_parameters = state.model.save_reference_parameters()
                reference_features = state.model.extract_features(x)
                reference_ntk = None
                if "ntk_change" in metrics:
                    try:
                        from ..diagnostics.curvature import ntk_matrix
                    except ImportError:
                        from diagnostics.curvature import ntk_matrix
                    reference_ntk = ntk_matrix(state.model, x).detach()

                losses = train_on_task(
                    state.model, state.optimizer, state.training_method, x, y, loss_fn, config["num_epochs_per_task"]
                )
                diagnostics = collect_diagnostics(
                    state.model,
                    x,
                    y,
                    loss_fn,
                    metrics,
                    reference_parameters=reference_parameters,
                    reference_features=reference_features,
                    reference_ntk=reference_ntk,
                )
                store_task_results(state.results, losses, diagnostics, metrics)

        for method_name, state in method_states.items():
            run_results[method_name].append(convert_results_to_numpy(state.results))

    stacked = {
        method_name: stack_runs(method_runs, metrics)
        for method_name, method_runs in run_results.items()
    }
    return stacked, task_runs, initial_state
