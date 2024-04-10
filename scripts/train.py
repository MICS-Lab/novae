from __future__ import annotations

import argparse

import lightning as L
import pandas as pd
import scanpy as sc
import yaml
from anndata import AnnData
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import novae
from novae import log
from novae.monitor import ComputeSwavOutputsCallback, EvalCallback, LogDomainsCallback


def load_datasets(relative_path: str) -> list[AnnData]:
    data_dir = novae.utils.repository_path() / "data"
    full_path = data_dir / relative_path

    if full_path.is_file():
        log.info(f"Loading one adata: {full_path}")
        return sc.read_h5ad(full_path)

    if ".h5ad" in relative_path:
        all_paths = list(map(str, data_dir.rglob(relative_path)))
    else:
        all_paths = list(map(str, full_path.rglob("*.h5ad")))

    log.info(f"Loading {len(all_paths)} adata(s): {', '.join(all_paths)}")
    return [sc.read_h5ad(path) for path in all_paths]


def get_training_mode(config: dict) -> str:
    if "model_kwargs" not in config or "swav" not in config["model_kwargs"]:
        return "swav"
    if config["model_kwargs"]["swav"]:
        return "swav"
    return "shuffle"


def main(args: argparse.Namespace) -> None:
    with open(novae.utils.repository_path() / "config" / args.config, "r") as f:
        config: dict = yaml.safe_load(f)
        config_flat = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]
        log.info(f"Using config {args.config}:\n{config}")

    adata = load_datasets(config["data"]["train_dataset"])

    mode = get_training_mode(config)
    log.info(f"Training mode: {mode}")

    wandb_logger = WandbLogger(
        log_model="all", project=f"novae_{mode}", **config.get("wandb_init_kwargs", {})
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
