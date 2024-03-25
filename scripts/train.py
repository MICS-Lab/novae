from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import scanpy as sc
import yaml
from anndata import AnnData
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from graph_lr import GraphLR, log
from graph_lr.utils import repository_path
from graph_lr.utils.monitor import log_domains_plots


def load_datasets(data_path: Path) -> list[AnnData]:
    if data_path.is_dir():
        all_paths = list(map(str, data_path.rglob("*.h5ad")))
        log.info(f"Loading {len(all_paths)} adatas: {', '.join(all_paths)}")
        return [sc.read_h5ad(path) for path in all_paths]
    else:
        log.info(f"Loading one adata: {data_path}")
        return sc.read_h5ad(data_path)


def main(args):
    repo_path = repository_path()

    with open(repo_path / "config" / args.config, "r") as f:
        config: dict = yaml.safe_load(f)
        log.info(f"Using config {args.config}:\n{config}")

    adata = load_datasets(repo_path / "data" / config["data"]["train_dataset"])

    is_swav = config["mode"] == "swav"
    log.info(f"Training mode: {config['mode']}")

    wandb_logger = None
    if config["use_wandb"]:
        wandb_logger = WandbLogger(log_model="all", project=f"graph_lr_{config['mode']}")

    model = GraphLR(adata, is_swav, **config["model_kwargs"])

    callbacks = [ModelCheckpoint(monitor="loss_epoch")]

    trainer = L.Trainer(logger=wandb_logger, callbacks=callbacks, **config["trainer_kwargs"])
    trainer.fit(model)

    model.swav_head.hierarchical_clustering()

    adata_eval = load_datasets(repo_path / "data" / config["data"]["eval_dataset"])
    model.swav_classes(adata_eval)
    log_domains_plots(model, adata_eval)


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
