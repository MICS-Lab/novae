# Pre-training scripts

These scripts are used to train and monitor Novae with Weight & Biases. To see the actualy source code, refer to the `novae` directory.

## Setup

For monitoring, `novae` must be installed with the `monitor` extra, for instance via pip:

```sh
pip install -e ".[monitor]"
```

## Usage

In the `data` directory, make sure to have `.h5ad` files. You can use the downloading scripts to get public data.
The corresponding `AnnData` object should contain raw counts, or preprocessed with `normalize_total` and `log1p`.

### Normal training

Choose a config inside the `config` directory.

```sh
python train.py --config <NAME>.yaml
```

### Sweep training

Choose a sweep config inside the `sweep` directory.

Inside the `scripts` directory, initialize the sweep with:
```sh
wandb sweep --project novae_swav sweep/<NAME>.yaml
```

Run the sweep with:
```sh
wandb agent <SWEEP_ID> --count 1
```

### SLURM usage

In the `slurm` directory:
- `train.sh` / `train_cpu.sh` for training
- `download.sh` to download public data
- `agent.sh SWEEP_ID COUNT` to run agents (where SWEEP_ID comes from `wandb sweep --project novae_swav sweep/<NAME>.yaml`)
