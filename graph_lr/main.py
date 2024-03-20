from __future__ import annotations

import argparse

import pytorch_lightning as pl
import scanpy as sc
import yaml
from anndata import AnnData
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from . import GraphLR, log
from .utils import repository_path


def init_project(args) -> tuple[AnnData, dict, WandbLogger]:
    repo_path = repository_path()

    with open(repo_path / "config" / args.config, "r") as f:
        config: dict = yaml.safe_load(f)
        log.info(f"Using config {args.config}:\n{config}")

    log.info(f"Loading data {args.dataset}")
    adata = sc.read_h5ad(repo_path / "data" / args.dataset)

    mode = "swav" if config["swav"] else "shuffle"
    log.info(f"Training mode: {mode}")

    if config["use_logger"]:
        wandb_logger = WandbLogger(log_model="all", project=f"graph_lr_{mode}")
    else:
        wandb_logger = None

    return adata, config, wandb_logger


def main(args):
    adata, config, wandb_logger = init_project(args)

    model = GraphLR(adata, config["swav"], **config["model_kwargs"])

    callbacks = [ModelCheckpoint(monitor="loss_epoch")]

    trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, **config["trainer_kwargs"])
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Name of the config to be used",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Name (or path) of the dataset to train on (relative to the 'data' directory)",
    )

    main(parser.parse_args())
