# Value-Prediction Plasticity Experiment

This implements a simple continual-learning plasticity test:

- states are one-hot vectors for `s in {1, ..., 100}`
- targets for each task are new random value functions sampled from `N(0, 1)`
- the continual network keeps its weights across tasks
- the fresh control reinitializes the same architecture and optimizer for every task
- both conditions train on the exact same target for exactly 500 updates per task

## Submit Parallel Seeds On Compute Canada

`cc_exp.sh` is a Slurm job-array script. By default it submits seeds `0..4`, one seed per array task:

```bash
sbatch cc_exp.sh
```

Run more seeds by overriding the array range:

```bash
sbatch --array=0-19 cc_exp.sh
```

Pass experiment options after the script name:

```bash
sbatch --array=0-19 cc_exp.sh --num-tasks 200
sbatch --array=0-19 cc_exp.sh --optimizer adam --learning-rate 0.001
sbatch --array=0-19 cc_exp.sh --reset-continual-optimizer-per-task
```

Each array task writes one seed to:

```text
results/value_prediction_plasticity/seed_<seed>/
```

Slurm logs are written to `slurm/logs/`.

## Aggregate Results

After all array jobs finish, combine the per-seed CSVs and regenerate the multi-seed plot:

```bash
source ~/ENV/rl/bin/activate
python aggregate_results.py
```

Combined outputs are written to:

```text
results/value_prediction_plasticity/combined/
```

## Run Locally

```bash
source ~/ENV/rl/bin/activate
python main.py
```

The default uses full-batch SGD without momentum to avoid optimizer-state history confounding the plasticity measurement.
