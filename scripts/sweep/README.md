## Wandb sweep configuration

This is the structure of a wandb config file:

```yaml
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

command:
  - python
  - train.py
  - --config
  - swav_cpu_0.yaml # name of the config under the scripts/config directory
  - --sweep
```