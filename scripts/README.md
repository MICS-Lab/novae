# Pre-training scripts

These scripts are used to pretrain and monitor Novae with Weight & Biases. To see the actualy source code of Novae, refer to the `novae` directory.

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
python -m scripts.train --config <NAME>.yaml
```

### Sweep training

Choose a sweep config inside the `sweep` directory.

Inside the `scripts` directory, initialize the sweep with:
```sh
wandb sweep --project novae sweep/<NAME>.yaml
```

Run the sweep with:
```sh
wandb agent <SWEEP_ID> --count 1
```

### Slurm usage

⚠️ Warning: the scripts below are specific to one cluster (Flamingo, Gustave Roussy). You'll need to update the `.sh` scripts according to your cluster.

In the `slurm` directory:
- `train.sh` / `train_cpu.sh` for training
- `download.sh` to download public data
- `sbatch agent.sh SWEEP_ID COUNT` to run agents (where SWEEP_ID comes from `wandb sweep --project novae sweep/<NAME>.yaml`)

E.g., on ruche:
```sh
module load anaconda3/2024.06/gcc-13.2.0 && source activate novae
wandb sweep --project novae sweep/gpu_ruche.yaml

cd ruche
sbatch agent.sh SWEEP_ID
```
