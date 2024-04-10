from __future__ import annotations

import argparse

import lightning as L
import pandas as pd
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import novae
from novae import log
from novae.monitor import ComputeSwavOutputsCallback, EvalCallback, LogDomainsCallback


def get_training_mode(config: dict) -> str:
    if "model_kwargs" not in config or "swav" not in config["model_kwargs"]:
        return "swav"
    if config["model_kwargs"]["swav"]:
        return "swav"
    return "shuffle"


def main(args: argparse.Namespace) -> None:
    with open(novae.utils.repository_root() / "config" / args.config, "r") as f:
        config: dict = yaml.safe_load(f)
        config_flat = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]
        log.info(f"Using config {args.config}:\n{config}")

    adata = novae.utils._load_dataset(config["data"]["train_dataset"])

    mode = get_training_mode(config)
    log.info(f"Training mode: {mode}")

    wandb_logger = WandbLogger(
        save_dir=novae.utils.wandb_log_dir(),
        log_model="all",
        project=f"novae_{mode}",
        **config.get("wandb_init_kwargs", {}),
    )
    wandb_logger.experiment.config.update(config_flat)

    model = novae.Novae(adata, **config.get("model_kwargs", {}))

    callbacks = [
        ModelCheckpoint(monitor="train/loss_epoch"),
        ComputeSwavOutputsCallback(),
        LogDomainsCallback(),
        EvalCallback(),
    ]

    trainer = L.Trainer(
        logger=wandb_logger, callbacks=callbacks, **config.get("trainer_kwargs", {})
    )
    trainer.fit(model, datamodule=model.datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Name of the YAML config to be used for training",
    )

    main(parser.parse_args())
