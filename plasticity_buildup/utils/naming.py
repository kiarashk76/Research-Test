import os


def _dims(value):
    if isinstance(value, int):
        value = [value]
    return "-".join(str(dim) for dim in value)


def make_experiment_directory(config, root="outputs"):
    x_label = "xreset" if config["reset_x"] else "xfixed"
    path = os.path.join(
        root,
        f"{config['dataset']}_{x_label}",
        f"in{config['input_dim']}_h{_dims(config['hidden_dims'])}_out{config['output_dim']}",
        f"samples{config['num_samples']}_tasks{config['num_tasks']}_epochs{config['num_epochs_per_task']}",
        f"{config.get('optimizer', 'adam')}_lr{config['learning_rate']}",
    )
    os.makedirs(path, exist_ok=True)
    return path


def experiment_metadata(config, task_file, initial_model_file):
    architecture = f"in{config['input_dim']}_h{_dims(config['hidden_dims'])}_out{config['output_dim']}"
    identifier = "|".join([
        config["dataset"], architecture, str(config["num_samples"]), str(config["num_tasks"]),
        str(config["reset_x"]), str(task_file), str(initial_model_file),
    ])
    return {"dataset": config["dataset"], "architecture": architecture, "task_file": task_file, "initial_model_file": initial_model_file, "identifier": identifier}
