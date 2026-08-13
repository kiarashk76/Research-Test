import copy

import torch

try:
    from tqdm import tqdm
except ImportError:  # Keep experiments runnable in minimal environments.
    def tqdm(iterable, **kwargs):
        return iterable

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
    optimizer_name = config.get("optimizer", "adam").lower()
    parameters = model.parameters()
    learning_rate = config["learning_rate"]
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if optimizer_name in {"sgd_momentum", "sgd+momentum", "momentum"}:
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
        )
    raise ValueError("optimizer must be one of: 'sgd', 'sgd_momentum', or 'adam'")


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
        training_method = make_training_method(
            method_config.get("method_type", "backprop"),
            model,
            optimizer,
            method_config,
            initial_state=initial_state,
        )
        states[method_name] = MethodState(model, optimizer, training_method, create_empty_results(metrics))
    return states


def _statistics_to_python(value):
    if isinstance(value, dict):
        return {name: _statistics_to_python(item) for name, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return value


def run_method_on_tasks(config, method_name, method_config, metrics, tasks, initial_state, run_seed, return_statistics=False):
    """Run exactly one method on one saved run without generating any data."""
    set_seed(run_seed)
    states = _make_method_states({method_name: method_config}, metrics, config, initial_state)
    state = states[method_name]
    loss_fn = make_loss_function(config)
    method_statistics = []
    task_iterator = tqdm(tasks, desc=f"{method_name} | seed {run_seed}", unit="task")
    for task in task_iterator:
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
        method_statistics.append(_statistics_to_python(state.training_method.get_statistics()))
    result = convert_results_to_numpy(state.results)
    if return_statistics:
        return result, method_statistics
    return result


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
        for task in tqdm(tasks, desc=f"run {run_index + 1}/{len(task_runs)}", unit="task"):
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
