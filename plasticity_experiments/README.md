# Plasticity Experiments

Small Python/PyTorch research codebase for studying loss of neural-network
plasticity in supervised learning and reinforcement learning.

The code is intentionally direct. Experiments are meant to be readable by
opening the relevant script, editing a few lines, and rerunning.

## Structure

```text
plasticity_experiments/
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
├── networks/
├── supervised_tasks/
├── rl_envs/
├── supervised_models/
├── rl_agents/
├── interventions/
├── experiments/
├── analysis/
└── results/
```

## Install

```bash
cd plasticity_experiments
conda activate rl
pip install -r requirements.txt
```

## Run

Supervised experiment:

```bash
cd plasticity_experiments
python main.py --config config.yaml --experiment supervised
```

RL experiment:

```bash
cd plasticity_experiments
python main.py --config config.yaml --experiment rl
```

Each run creates a directory like:

```text
results/supervised_random_20260729_143000/
├── config.yaml
├── seed_0.csv
├── seed_1.csv
├── seed_2.csv
├── task_data/  # optional, only when task_data.enabled is true
├── artifacts/  # optional, only when artifacts.enabled is true
└── analysis/   # supervised runs create this automatically
```

## Interventions

Choose the intervention in `config.yaml`:

```yaml
intervention:
  type: random  # none, fresh, random, dormant, or small_gradient
  reset_fraction: 0.05
  dormant_threshold: 0.01
  interval: 100
```

Intervention types:

- `none`: one continual network carries weights across tasks with no reset.
- `fresh`: reinitialize the network at every task boundary.
- `random`: reset a random fraction of hidden neurons.
- `dormant`: reset neurons with low average hidden activation.
- `small_gradient`: reset neurons with the smallest gradient scores.

Resetting a hidden neuron reinitializes its incoming weights and bias, resets its
outgoing connections, and clears matching optimizer state when an optimizer is
provided.

## Supervised Experiment

`supervised_tasks/random_targets.py` defines fixed inputs with deterministic
task-dependent random targets. `experiments/supervised_experiment.py` trains one
model trajectory per run. Use `intervention.type: fresh` when you want the fresh
baseline.

The CSV records seed, task, update, global update, model type, train loss,
evaluation loss, intervention type, number of reset neurons, and optional
artifact path.

For supervised runs:

- `train_loss` is MSE on the sampled training batch for that update.
- `eval_loss` is MSE on the full fixed evaluation input set for the current task.
- Final-performance plots use the last recorded `eval_loss` within each task.

## Task Data

Set this block in `config.yaml` to save the generated supervised datasets:

```yaml
task_data:
  enabled: true
  save_inputs: true
  save_targets: true
```

When enabled, each supervised seed writes:

```text
<run_dir>/task_data/seed_<seed>/
├── metadata.pt
├── inputs.pt
└── targets/
    ├── task_0000_targets.pt
    ├── task_0001_targets.pt
    └── ...
```

The inputs are shared across tasks. Each task target file contains the target
tensor for those same inputs on that task. Load them with:

```python
import torch

x = torch.load("task_data/seed_0/inputs.pt", map_location="cpu")
y0 = torch.load("task_data/seed_0/targets/task_0000_targets.pt", map_location="cpu")
y1 = torch.load("task_data/seed_0/targets/task_0001_targets.pt", map_location="cpu")
```

## RL Experiment

`rl_envs/switching_mdp.py` defines a one-step contextual bandit using Gymnasium.
Each task changes which action is rewarding for each state. `rl_agents/dqn_agent.py`
contains a minimal DQN-style agent with replay, a target network, and
epsilon-greedy action selection.

`experiments/rl_experiment.py` trains:

- one DQN agent trajectory per run;
- use `intervention.type: fresh` to reinitialize the agent at each task.

The CSV records seed, task, environment step, global step, agent type,
evaluation return, TD loss, epsilon, intervention type, reset count, and
optional artifact path.

For RL runs:

- `td_loss` is the DQN Bellman target MSE on the sampled replay batch.
- `eval_return` is average reward over evaluation episodes on the current task,
  using greedy actions with exploration turned off.
- Final-performance plots use the last recorded `eval_return` within each task.

## Eval Artifacts

Set this block in `config.yaml` when you want tensor dumps at every eval step:

```yaml
artifacts:
  enabled: true
  save_weights: true
  save_gradients: true
  save_activations: true
  max_activation_examples: null
```

When enabled, each eval row in the seed CSV gets an `artifact_path` pointing to a
`.pt` file under:

```text
<run_dir>/artifacts/seed_<seed>/<intervention_type>/
```

Each artifact is a pickle-based PyTorch dump created with `torch.save`. It
contains:

- `metadata`: seed, actor, task, step, global step, metric values, intervention;
- `weights`: model `state_dict`;
- `extra_state_dicts`: for RL, the DQN target-network `state_dict`;
- `gradients`: parameter gradients, when available;
- `activations`: eval inputs, hidden-layer activations, and network outputs.

For supervised runs, the artifact is computed on the full current-task dataset at
that eval step:

- `mse` / `eval_loss`: full-dataset MSE;
- `weights`: model weights at that eval step;
- `activations`: full-dataset inputs, targets, hidden activations, and outputs
  from a forward pass;
- `gradients`: gradients from `backward()` on the full-dataset MSE.

Load one with:

```python
import torch

artifact = torch.load("path/to/artifact.pt", map_location="cpu")
```

Full activation dumps can be large. Set `max_activation_examples` to an integer
to store activations for only the first N eval examples/states.

## Analysis

Analysis is currently focused on supervised runs and reads the saved seed CSV
files. It writes four seed-averaged metrics/plots:

- `latest_eval_mse_by_task`: y-axis is latest eval MSE in each task, x-axis is
  task number.
- `eval_mse_by_eval_step`: y-axis is eval MSE, x-axis is eval step, with gray
  vertical lines marking task changes.
- `eval_mse_auc_by_task`: y-axis is trapezoidal AUC for eval MSE within each
  task, x-axis is task number.
- `steps_to_mse_threshold_by_task`: y-axis is the first training step within a
  task where eval MSE reaches the configured threshold, x-axis is task number.
  The threshold comes from this plot's nested config.

```yaml
analysis:
  latest_eval_mse_by_task: true
  eval_mse_by_eval_step: true
  eval_mse_auc_by_task: true
  steps_to_mse_threshold_by_task:
    enabled: true
    mse_threshold: 0.01
```

Set any plot flag to `false` to skip that metric/plot during automatic analysis
at the end of `main.py`. For the threshold plot, set `enabled: false`.

Example:

```bash
python analyze.py results/supervised_random_20260729_143000
```

`main.py --experiment supervised` runs this analysis automatically after all
seeds finish. Single-run outputs are written under:

```text
<run_dir>/analysis/
```

The exact output files are:

```text
latest_eval_mse_by_task.csv
latest_eval_mse_by_task.png
eval_mse_by_eval_step.csv
eval_mse_by_eval_step.png
eval_mse_auc_by_task.csv
eval_mse_auc_by_task.png
steps_to_mse_threshold_by_task.csv
steps_to_mse_threshold_by_task.png
```

Compare multiple runs in one plot:

```bash
python analyze.py \
  results/supervised_none_20260729_143000 \
  results/supervised_random_20260729_144000 \
  results/supervised_dormant_20260729_145000
```

Alternatively, edit `RUN_DIRS` near the top of `analyze.py` and run:

```bash
python analyze.py
```

Comparison plots include one curve per selected run/intervention. Multi-run
outputs are written to `<runs_parent>/comparison_analysis/` unless `--output-dir`
is provided. `analyze.py` clears old analysis CSV/PNG files in the output folder
before writing these files, so stale plots from older analysis code do not
linger.

Python usage:

```python
from analysis.metrics import (
    eval_mse_auc_by_task,
    eval_mse_by_eval_step,
    latest_eval_mse_by_task,
    prepare_supervised_eval_df,
    steps_to_mse_threshold_by_task,
)

df = prepare_supervised_eval_df("results/supervised_random_20260729_143000")
latest = latest_eval_mse_by_task(df)
curve = eval_mse_by_eval_step(df)
auc = eval_mse_auc_by_task(df)
steps = steps_to_mse_threshold_by_task(df, mse_threshold=0.01)
```

## Adding New Ideas

- New network: add a file in `networks/` and instantiate it in the model or agent.
- New supervised task: add a file in `supervised_tasks/` with `set_task`,
  `sample_batch`, and `get_evaluation_data`.
- New RL environment: add a Gymnasium env in `rl_envs/` with `set_task`.
- New supervised model wrapper: add it in `supervised_models/`.
- New RL agent: add it in `rl_agents/`.
- New reset method: add a plain function in `interventions/` and one `elif` in the
  experiment script where interventions are applied.

The point is short feedback cycles and controlled comparisons, not a general
experiment framework.
