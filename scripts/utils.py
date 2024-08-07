from __future__ import annotations

import argparse
from collections import defaultdict

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from anndata import AnnData
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import novae
import wandb
from novae import log
from novae._constants import Keys
from novae.monitor import jensen_shannon_divergence, mean_fide_score, mean_svg_score
from novae.monitor.callback import (
    LogProtoCovCallback,
    LogTissuePrototypeWeights,
    ValidationCallback,
)


def init_wandb_logger(config: dict[str, str | dict]) -> WandbLogger:
    wandb.init(project=config.get("project", "novae"), **config["wandb_init_kwargs"])

    if config.get("sweep"):
        sweep_config = dict(wandb.config)
        if "lr" in sweep_config:
            config["fit_kwargs"] = config["fit_kwargs"] | {"lr": wandb.config.lr}
            del sweep_config["lr"]
        config["model_kwargs"] = config["model_kwargs"] | sweep_config

    log.info(f"Full config:\n{config}")

    assert "slide_key" not in config["model_kwargs"], "For now, please provide one adata per file."

    wandb_logger = WandbLogger(
        save_dir=novae.utils.wandb_log_dir(),
        log_model="all",
        project=config.get("project", "novae"),
    )

    config_flat = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]
    wandb_logger.experiment.config.update(config_flat)

    return wandb_logger


def _get_hardware_kwargs(config: dict) -> dict:
    return {
        "num_workers": config["fit_kwargs"].get("num_workers", 0),
        "accelerator": config["fit_kwargs"].get("accelerator", "cpu"),
    }


def get_callbacks(config: dict, adatas_val: list[AnnData] | None) -> list[L.Callback] | None:
    if config["wandb_init_kwargs"].get("mode") == "disabled":
        return None

    validation_callback = ValidationCallback(adatas_val, **_get_hardware_kwargs(config))

    if config["sweep"]:
        return [validation_callback]

    return [
        validation_callback,
        ModelCheckpoint(monitor="metrics/val_heuristic", mode="max"),
        LogProtoCovCallback(),
        LogTissuePrototypeWeights(),
    ]


def read_config(args: argparse.Namespace) -> dict:
    with open(novae.utils.repository_root() / "scripts" / "config" / args.config, "r") as f:
        config = defaultdict(dict, yaml.safe_load(f))
        config["sweep"] = args.sweep
        return config


def post_training(model: novae.Novae, adatas: list[AnnData], config: dict):
    if config["post_training"].get("save_umap") or config["post_training"].get("save_metrics"):
        model.compute_representation(**_get_hardware_kwargs(config))

    if config["post_training"].get("save_umap"):
        _save_umap(model, config)

    if config["post_training"].get("save_result"):
        _save_result(model, config)

    if config["post_training"].get("save_metrics"):
        for k in [5, 7, 10, 15]:
            obs_key = model.assign_domains(k=k)
            jsd = jensen_shannon_divergence(adatas, obs_key)
            fide = mean_fide_score(adatas, obs_key, n_classes=k)
            svg = mean_svg_score(adatas, obs_key)
            log.info(f"[{k=}] JSD: {jsd}, FIDE: {fide}, SVG: {svg}")


def _save_umap(model: novae.Novae, config: dict):
    obs_key = model.assign_domains(k=config["post_training"]["save_umap"])
    model.batch_effect_correction()

    for adata in model.adatas:
        if "novae_tissue" in adata.uns:
            adata.obs["tissue"] = adata.uns["novae_tissue"]

    latent_conc = np.concat([adata.obsm[Keys.REPR_CORRECTED] for adata in model.adatas], axis=0)
    obs_conc = pd.concat([adata.obs for adata in model.adatas], axis=0)
    adata_conc = AnnData(obsm={Keys.REPR_CORRECTED: latent_conc}, obs=obs_conc)
    n_obs_th = 500_000
    if adata_conc.n_obs > n_obs_th:
        sc.pp.subsample(adata_conc, n_obs=n_obs_th)
    sc.pp.neighbors(adata_conc, use_rep=Keys.REPR_CORRECTED)
    sc.tl.umap(adata_conc)
    colors = [obs_key]
    for key in ["tissue", "technology"]:
        if key in adata_conc.obs:
            colors.append(key)
    sc.pl.umap(adata_conc, color=colors, show=False)
    wandb.log({"umap": wandb.Image(plt)})
    plt.close()


def _save_result(model: novae.Novae, config: dict):
    model.compute_representation(**_get_hardware_kwargs(config))
    for k in [5, 7, 10, 15]:
        model.assign_domains(k=k)
    res_dir = novae.utils.repository_root() / "data" / "results" / config["post_training"]["save_result"]
    res_dir.mkdir(parents=True, exist_ok=True)

    for adata in model.adatas:
        out_path = res_dir / f"{id(adata)}.h5ad"
        log.info(f"Writing adata file to {out_path}: {adata}")
        adata.write_h5ad(out_path)
