"""
Novae model training with Weight & Biases monitoring
This is **not** the actual Novae source code. Instead, see the `novae` directory
"""

from __future__ import annotations

import argparse

import lightning as L
import pandas as pd
import torch
import yaml
from anndata import AnnData
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import novae
import wandb
from novae import log
from novae.monitor.callback import LogProtoCovCallback, ValidationCallback


def train(adatas: list[AnnData], config: dict, sweep: bool = False, adatas_val: list[AnnData] | None = None):
    """Train Novae on adata. This function can be used inside wandb sweeps.

    If `sweep is True`, the sweep parameters will update the `model_kwargs` from the YAML config.

    Args:
        adatas: List of AnnData objects
        config: The config dict corresponding to a YAML file inside the `config` directory
        sweep: Whether we are running wandb sweeps or not
        adatas_val: List of AnnData objects used for validation
    """
    wandb.init(project="novae", **config.get("wandb_init_kwargs", {}))

    if sweep:
        config["model_kwargs"] = config.get("model_kwargs", {}) | dict(wandb.config)
    log.info(f"Full config:\n{config}")

    assert "slide_key" not in config.get(
        "model_kwargs", {}
    ), "'slide_key' not supported in model_kwargs yet. Provide one adata per file."

    wandb_logger = WandbLogger(save_dir=novae.utils.wandb_log_dir(), log_model="all", project="novae")

    config_flat = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]
    wandb_logger.experiment.config.update(config_flat)
    callbacks = _get_callbacks(config, sweep, adatas_val)

    model = novae.Novae(adatas, **config.get("model_kwargs", {}))

    if config.get("compile"):
        log.info("Compiling the model")
        model: novae.Novae = torch.compile(model)

    model.fit(logger=wandb_logger, callbacks=callbacks, **config.get("trainer_kwargs", {}))

    if config.get("save_result"):
        _save_result(model, config)


def _save_result(model: novae.Novae, config: dict):
    model.compute_representation(**_get_hardware_kwargs(config))
    for k in [5, 7, 10, 15]:
        model.assign_domains(k=k)
    res_dir = novae.utils.repository_root() / "data" / "results" / config["save_result"]
    res_dir.mkdir(parents=True, exist_ok=True)

    for adata in model.adatas:
        print(adata.obs["novae_leaves"])
        del adata.obs["novae_leaves"]
        out_path = res_dir / f"{id(adata)}.h5ad"
        log.info(f"Writing adata file to {out_path}: {adata}")
        adata.write_h5ad(out_path)


def _get_hardware_kwargs(config: dict) -> dict:
    num_workers = config.get("trainer_kwargs", {}).get("num_workers")
    accelerator = config.get("trainer_kwargs", {}).get("accelerator")
    return {"num_workers": num_workers, "accelerator": accelerator}


def _get_callbacks(config: dict, sweep: bool, adatas_val: list[AnnData] | None) -> list[L.Callback] | None:
    if config.get("wandb_init_kwargs", {}).get("mode") == "disabled":
        return None

    callbacks = [ValidationCallback(adatas_val, **_get_hardware_kwargs(config))]

    if sweep:
        return callbacks

    callbacks.extend([ModelCheckpoint(monitor="train/loss_epoch"), LogProtoCovCallback()])

    return callbacks


def _read_config(name: str) -> dict:
    with open(novae.utils.repository_root() / "scripts" / "config" / name, "r") as f:
        return yaml.safe_load(f)


def main(args: argparse.Namespace) -> None:
    """
    Load the dataset, read the config, and run training (with or without sweeps)
    """
    config = _read_config(args.config)

    train_dataset_path = config["data"]["train_dataset"]
    adatas = novae.utils._load_dataset(train_dataset_path)

    val_dataset_path = config["data"].get("val_dataset")
    adatas_val = novae.utils._load_dataset(val_dataset_path) if val_dataset_path else None

    train(adatas, config, sweep=args.sweep, adatas_val=adatas_val)


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
