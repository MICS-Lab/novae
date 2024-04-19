"""
Novae model training with Weight&Biases monitoring
This is **not** the actual Novae source code. Instead, see the `novae` directory
"""

from __future__ import annotations

import argparse

import lightning as L
import pandas as pd
import yaml
from anndata import AnnData
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import novae
import wandb
from novae import log, monitor


def train(adata: AnnData, config: dict, project: str, sweep: bool = False):
    """Train Novae on adata. This function can be used inside wandb sweeps.

    If `sweep is True`, the sweep parameters will update the `model_kwargs` from the YAML config.

    Args:
        adata: One or multiple AnnData objects
        config: The config dict corresponding to a YAML file inside the `config` directory
        project: Name of the wandb project
        sweep: Whether we are running wandb sweeps or not
    """
    wandb.init(project=project, **config.get("wandb_init_kwargs", {}))

    if sweep:
        config["model_kwargs"] = config.get("model_kwargs", {}) | dict(wandb.config)
    log.info(f"Full config:\n{config}")

    wandb_logger = WandbLogger(save_dir=novae.utils.wandb_log_dir(), log_model="all", project=project)

    config_flat = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]
    wandb_logger.experiment.config.update(config_flat)

    model = novae.Novae(adata, **config.get("model_kwargs", {}))

    callbacks = [ModelCheckpoint(monitor="train/loss_epoch")]

    if not sweep:
        callbacks.extend(
            [
                monitor.ComputeSwavOutputsCallback(),
                monitor.LogDomainsCallback(),
                monitor.EvalCallback(),
                monitor.LogLatent(),
                monitor.LogProtoCovCallback(),
            ]
        )

    trainer = L.Trainer(logger=wandb_logger, callbacks=callbacks, **config.get("trainer_kwargs", {}))
    trainer.fit(model, datamodule=model.datamodule)


def _read_config(name: str) -> dict:
    with open(novae.utils.repository_root() / "scripts" / "config" / name, "r") as f:
        return yaml.safe_load(f)


def _get_training_mode(config: dict) -> str:
    if "model_kwargs" not in config or "swav" not in config["model_kwargs"]:
        return "swav"
    if config["model_kwargs"]["swav"]:
        return "swav"
    return "shuffle"


def main(args: argparse.Namespace) -> None:
    """
    Load the dataset, read the config, and run training (with or without sweeps)
    """
    config = _read_config(args.config)

    adata = novae.utils._load_dataset(config["data"]["train_dataset"])

    mode = _get_training_mode(config)
    project = f"novae_{mode}"
    log.info(f"Training mode: {mode}")

    train(adata, config, project, sweep=args.sweep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Fullname of the YAML config to be used for training (see under the `config` directory)",
    )
    parser.add_argument("-s", "--sweep", nargs="?", default=False, const=True, help="Whether it is a sweep or not")

    main(parser.parse_args())
