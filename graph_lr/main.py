from __future__ import annotations

import argparse

import pytorch_lightning as pl
import scanpy as sc
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from . import GraphCL
from .utils import repository_path


def main(args):
    data_path = repository_path() / "data" / args.path

    adata = sc.read_h5ad(data_path)

    swav = True
    slide_key = None  # "ID"

    mode = "swav" if swav else "shuffle"
    model = GraphCL(adata, swav, n_hops=2, heads=4, slide_key=slide_key)

    wandb_logger = WandbLogger(log_model="all", project=f"graph_lr_{mode}")

    callbacks = [ModelCheckpoint(monitor="loss_epoch")]

    trainer = pl.Trainer(
        max_epochs=40,
        accelerator="cpu",
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to the dataset to train on (relative to the 'data' directory)",
    )

    main(parser.parse_args())
