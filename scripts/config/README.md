# Config

These `.yaml` files are used when training with Weight & Biases with the `train.py` script.

## Description

This is a minimal YAML config used to explain its structure. The dataset names are relative to the `data` directory. If a directory, loads every `.h5ad` files inside it. Can also be a file, or a file pattern.

```yaml
data:
  train_dataset: merscope # training dataset name
  val_dataset: xenium # eval dataset name

model_kwargs: # Novae model kwargs
  heads: 4

fit_kwargs: # Trainer kwargs (from Lightning)
  max_epochs: 3
  log_every_n_steps: 10
  accelerator: "cpu"

wandb_init_kwargs: # wandb.init kwargs
  mode: "online"
```
