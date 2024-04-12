# Pre-training scripts

These scripts are used to train and monitor Novae with Weight & Biases. To see the actualy source code, refer to the `novae` directory.

## Setup

For monitoring, `novae` must be installed with the `monitor` extra, for instance via pip:

```sh
pip install -e ".[monitor]"
```

## Usage

Choose a config inside the `config` directory at the root of the project.

```sh
python scripts/train.py --config <NAME>.yaml
```
