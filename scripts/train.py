from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import scanpy as sc
import yaml
from anndata import AnnData
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import novae
from novae import log
from novae.monitor import ComputeSwavOutputsCallback, EvalCallback, LogDomainsCallback


def load_datasets(data_dir: Path, relative_path: str) -> list[AnnData]:
    full_path = data_dir / relative_path

    if full_path.is_file():
        log.info(f"Loading one adata: {full_path}")
        return sc.read_h5ad(full_path)

    if ".h5ad" in relative_path:
        all_paths = list(map(str, data_dir.rglob(relative_path)))
    else:
        all_paths = list(map(str, full_path.rglob("*.h5ad")))

    log.info(f"Loading {len(all_paths)} adatas: {', '.join(all_paths)}")
    return [sc.read_h5ad(path) for path in all_paths]


def main(args):
    repo_path = novae.utils.repository_path()

    with open(repo_path / "config" / args.config, "r") as f:
        config: dict = yaml.safe_load(f)
        log.info(f"Using config {args.config}:\n{config}")

    adata = load_datasets(repo_path / "data", config["data"]["train_dataset"])

    is_swav = config["mode"] == "swav"
    log.info(f"Training mode: {config['mode']}")

    wandb_logger = None
    if config["use_wandb"]:
        wandb_logger = WandbLogger(log_model="all", project=f"novae_{config['mode']}")

    model = novae.Novae(adata, is_swav, **config["model_kwargs"])

    callbacks = [
        ModelCheckpoint(monitor="loss_epoch"),
        ComputeSwavOutputsCallback(),
        LogDomainsCallback(),
        EvalCallback(),
    ]

    trainer = L.Trainer(logger=wandb_logger, callbacks=callbacks, **config["trainer_kwargs"])
    trainer.fit(model)


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
