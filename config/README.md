# Config

These `.yaml` files are used when training with Weight & Biases (as in the `scripts` directory).

## Description

This is a minimal YAML config used to explain its structure. The dataset names are relative to the `data` directory. If a directory, loads every `.h5ad` files inside it. Can also be a file, or a file pattern.

```yaml
data:
  train_dataset: merscope # training dataset name
  eval_dataset: xenium # eval dataset name

model_kwargs: # Novae model kwargs
  heads: 4

trainer_kwargs: # Trainer kwargs (from Lightning)
  max_epochs: 3
  log_every_n_steps: 10
  accelerator: "cpu"

wandb_init_kwargs: # wandb.init kwargs
  mode: "online"
```

## Running wandb sweep

To run a wandb sweep, this has to be added to the YAML file:

```yaml
sweep:
  count: 2 # number of trials
  configuration:
    method: random # search strategy (grid/random/bayes)

    metric:
      goal: minimize
      name: train/loss_epoch # the metric to be optimized

    parameters: # parameters to hyperoptimize (kwargs arguments of Novae)
      heads:
        values: [1, 4, 8]
      lr:
        min: 0.0001
        max: 0.01
```