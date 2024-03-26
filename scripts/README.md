# Pre-training scripts

These scripts are used to train and monitor Novae with Weight & Biases. To see the actualy source code, refer to the `novae` directory.

## Setup

For monitoring, `novae` must be installed with de `dev` extra, for instance via Poetry: 

```sh
poetry install --all-extras
```

## Usage

Choose a config inside the `config` directory at the root of the project.

```
python scripts/train.py --config <NAME>.yaml
```
