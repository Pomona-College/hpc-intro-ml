---
title: "Hyperparameter Tuning"
teaching: 25
exercises: 20
---

:::::::::::::::::::::::::::::::::::::: questions
- How do you efficiently tune hyperparameters on HPC?
- What are the differences between grid search, random search, and Bayesian optimization?
- How do you parallelize hyperparameter sweeps with SLURM?
- How do you avoid wasting GPU-hours on bad runs?
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Implement grid search and random search
- Use Bayesian optimization for efficient tuning
- Parallelize hyperparameter sweeps across multiple GPU jobs
- Use early stopping and pruning to save GPU-hours
::::::::::::::::::::::::::::::::::::::::::::::::

## Why Hyperparameter Tuning Is Different on HPC

A hyperparameter sweep that takes 100 model trainings is exactly the kind of workload HPC was built for. Each training is independent, GPUs are abundant, and SLURM job arrays make parallel submission trivial. The art is in choosing which 100 hyperparameter combinations to evaluate; brute-force grid search is rarely the right answer past one or two dimensions.

Three search strategies dominate practice: grid search (exhaustive on a discrete grid), random search (sample uniformly from continuous ranges), and Bayesian optimization (learn from past trials to pick promising next ones). Each has a sweet spot, and combining them with early stopping turns a week-long sweep into an overnight one.

## Hyperparameter Tuning Strategies

Hyperparameters are values set before training (learning rate, batch size, layers, etc.). Unlike weights, they are not learned. Their values can change model performance by an order of magnitude, which is why finding good ones is worth the GPU time.

### Strategy 1: Grid Search

Test all combinations of predefined values:

```python
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'hidden_units': [64, 128, 256]
}
# Total: 3 x 3 x 3 = 27 models to train
```

**Pros**: Systematic, guaranteed to find best in grid.
**Cons**: Exponential growth, slow for large spaces.

Grid search becomes intractable past three or four dimensions. With 5 hyperparameters at 5 values each, you have 3,125 trials. At 30 minutes per trial that is over 60 days of GPU time. Unless your search space is small, grid search is not the right tool.

### Strategy 2: Random Search

```python
import numpy as np
for trial in range(50):
    lr = np.random.choice(np.logspace(-4, -1, 100))
    bs = np.random.choice([16, 32, 64, 128])
    dp = np.random.choice(np.linspace(0.1, 0.5, 50))
    # Train model with these hyperparameters
```

**Pros**: More efficient than grid search, scales to many dimensions.
**Cons**: Each trial is independent; no learning from previous results.

Random search is surprisingly competitive with more sophisticated methods, especially in high-dimensional spaces, because most hyperparameters do not matter much and random sampling explores the few that do. Bergstra and Bengio (2012) showed that random search outperforms grid search consistently.

### Strategy 3: Bayesian Optimization

```python
from ray import tune

config = {
    'learning_rate': tune.loguniform(1e-4, 1e-1),
    'batch_size': tune.choice([16, 32, 64, 128]),
    'dropout_rate': tune.uniform(0.1, 0.5)
}

analysis = tune.run(
    train_model, config=config, num_samples=50,
    search_alg=tune.suggest.BayesOptSearch()
)
best_config = analysis.get_best_config(metric='val_loss', mode='min')
```

**Pros**: Most sample-efficient. Learns which regions of the hyperparameter space are promising and concentrates trials there.
**Cons**: Setup is more complex; benefits are largest when each trial is expensive.

Use Bayesian optimization when each model trial takes hours and you can afford 30 to 100 trials total. For very fast trials or very large numbers of trials, random search can be just as good and simpler.

## Parallel Search with SLURM Job Arrays

```bash
#!/bin/bash
#SBATCH --job-name=hyperparam_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=sweep_%a.log

module load miniconda3
conda activate pytorch_env

LEARNING_RATES=(0.001 0.005 0.01 0.02 0.05 0.1 0.15 0.2 0.25 0.3)
LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}

python3 train.py --learning_rate $LR --batch_size 32 --epochs 20
```

The `--array=0-9` tells SLURM to submit 10 copies of this script in parallel, each with a different `SLURM_ARRAY_TASK_ID` (0 through 9). The script picks the corresponding learning rate from the array. With 10 idle GPUs, all 10 trials run simultaneously and the whole sweep finishes in roughly the time of one trial instead of ten.

For a sweep over multiple parameters, the array-index-to-parameter mapping uses division and modulo:

```bash
#SBATCH --array=0-29   # 5 LRs * 6 BSs = 30 combos

LRS=(0.001 0.005 0.01 0.05 0.1)
BSS=(16 32 64 128 256 512)

lr_idx=$((SLURM_ARRAY_TASK_ID / ${#BSS[@]}))
bs_idx=$((SLURM_ARRAY_TASK_ID % ${#BSS[@]}))

LR=${LRS[$lr_idx]}
BS=${BSS[$bs_idx]}

python3 train.py --learning_rate $LR --batch_size $BS
```

Submit and gather results:

```bash
sbatch hyperparameter_sweep.sh
squeue -u $USER -l

# Gather results
for log in sweep_*.log; do
    echo "=== $log ==="
    grep "Final validation accuracy" $log
done
```

::::::::::::::::::::::::::::::::::::: callout

## Early stopping inside each trial

Even with random search, many trials are clearly bad after a few epochs. Killing them early saves GPU-hours that you can spend on more trials.

Two ways to implement: a simple validation-loss-plateau check inside `train.py`, or a pruning callback from a framework like Optuna or Ray Tune. The simple version is essentially:

```python
best_val = float('inf')
patience = 5
no_improve = 0
for epoch in range(max_epochs):
    val_loss = train_one_epoch(...)
    if val_loss < best_val:
        best_val = val_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping")
            break
```

A 100-trial sweep with 50% early-stopping savings completes in roughly the wall-clock of a 50-trial full sweep, but explores twice as much hyperparameter space.

::::::::::::::::::::::::::::::::::::::::::::::::

## Choosing the Search Range

The single most common hyperparameter-sweep mistake is searching the wrong range. A few rules of thumb:

Learning rate: search log-uniformly from 1e-5 to 1e-1. Linear spacing wastes most samples on the wrong order of magnitude.

Batch size: powers of 2 (16, 32, 64, ...). Memory and GPU efficiency favor these.

Dropout: linear from 0.1 to 0.5. Above 0.5 the model usually under-fits.

Weight decay: log-uniform from 1e-6 to 1e-2.

Number of layers: linear, 2 to 8. Past 8 you usually want a different architecture.

Document the range in your experiment log along with the search method. "Best LR was 0.003" is much less useful than "Searched log-uniform 1e-5 to 1e-1 with 30 random samples; best LR was 0.003".

::::::::::::::::::::::::::::::::::::: challenge

**Challenge: Hyperparameter Sweep**

Set up a SLURM array job that trains 10 models with 5 learning rates and 2 batch sizes. Collect results in a CSV file.

::::::::::::::::::::::::::::::::::::: solution

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=tune_%a.log

module load miniconda3
conda activate pytorch_env

python3 << 'EOF'
import os, numpy as np, csv

learning_rates = np.logspace(-4, -2, 5)
batch_sizes = [32, 64]
job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
lr = learning_rates[job_id // 2]
bs = batch_sizes[job_id % 2]

print(f"Training with lr={lr}, batch_size={bs}")
# ... training code ...

with open(f'/scratch/$USER/results.csv', 'a') as f:
    csv.writer(f).writerow([lr, bs, val_loss, val_accuracy])
EOF
```

After the sweep completes, sort `results.csv` by val_accuracy and inspect the top 3 rows. If they cluster at one end of the range, the optimum may lie outside your search range; expand and rerun.

::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

**Challenge: Pick a Strategy**

For each scenario, recommend grid / random / Bayesian and explain why in one sentence:

A. Tuning 2 hyperparameters (learning rate, weight decay) for a 5-minute model.

B. Tuning 8 hyperparameters for a 4-hour model on a constrained GPU budget.

C. Reproducing a published baseline whose paper specifies exact hyperparameters.

::::::::::::::::::::::::::::::::::::: solution

A. Grid search. Two dimensions, fast trials, easy to enumerate.

B. Bayesian optimization. Many dimensions and expensive trials make sample efficiency the priority.

C. None: use the published values directly. Tuning here would be wasted compute and risks divergence from the baseline.

::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints
- Grid search is fine for 2 to 3 dimensions, intractable beyond that
- Random search is competitive in higher dimensions and scales linearly
- Bayesian optimization wins when each trial is expensive (hours)
- Search log-uniformly for learning rate and weight decay; linear for dropout
- Early stopping inside each trial multiplies the effective number of trials
- SLURM job arrays parallelize sweeps with a single submission
- After a sweep, check that the best values are not at the edge of the range; expand if so
::::::::::::::::::::::::::::::::::::::::::::::::
