from __future__ import annotations

import argparse

import lightning as L
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
from novae.monitor.log import log_plt_figure

from .config import Config


def init_wandb_logger(config: Config) -> WandbLogger:
    wandb.init(project=config.project, **config.wandb_init_kwargs)

    if config.sweep:
        sweep_config = dict(wandb.config)
        if "lr" in sweep_config:
            config.fit_kwargs["lr"] = wandb.config.lr
            del sweep_config["lr"]
        for key, value in sweep_config.items():
            config.model_kwargs[key] = value

    log.info(f"Full config:\n{config.model_dump()}")

    assert "slide_key" not in config.model_kwargs, "For now, please provide one adata per file."

    wandb_logger = WandbLogger(
        save_dir=novae.utils.wandb_log_dir(),
        log_model="all",
        project=config.project,
    )

    config_flat = pd.json_normalize(config.model_dump(), sep=".").to_dict(orient="records")[0]
    wandb_logger.experiment.config.update(config_flat)

    return wandb_logger


def _get_hardware_kwargs(config: Config) -> dict:
    return {
        "num_workers": config.fit_kwargs.get("num_workers", 0),
        "accelerator": config.fit_kwargs.get("accelerator", "cpu"),
    }


def get_callbacks(config: Config, adatas_val: list[AnnData] | None) -> list[L.Callback] | None:
    if config.wandb_init_kwargs.get("mode") == "disabled":
        return None

    callbacks = [ValidationCallback(adatas_val, **_get_hardware_kwargs(config))] if adatas_val else []

    if config.sweep:
        return callbacks

    return callbacks + [
        ModelCheckpoint(monitor="metrics/val_heuristic", mode="max"),
        LogProtoCovCallback(),
        LogTissuePrototypeWeights(),
    ]


def read_config(args: argparse.Namespace) -> Config:
    with open(novae.utils.repository_root() / "scripts" / "config" / args.config, "r") as f:
        config = yaml.safe_load(f)
        config = Config(**config, sweep=args.sweep)
        return config


def post_training(model: novae.Novae, adatas: list[AnnData], config: Config):
    keys_repr = ["log_umap", "log_metrics", "log_domains"]
    if any(getattr(config.post_training, key) for key in keys_repr):
        model.compute_representation(**_get_hardware_kwargs(config))

    if config.post_training.log_domains:
        for n_domains in config.post_training.n_domains:
            obs_key = model.assign_domains(n_domains=n_domains)
            novae.plot.domains(adatas, obs_key)
            log_plt_figure(f"domains_{n_domains=}")

    if config.post_training.save_h5ad:
        _save_h5ad(model, config)

    if config.post_training.log_metrics:
        for n_domains in config.post_training.n_domains:
            obs_key = model.assign_domains(n_domains=n_domains)
            jsd = jensen_shannon_divergence(adatas, obs_key)
            fide = mean_fide_score(adatas, obs_key, n_classes=n_domains)
            svg = mean_svg_score(adatas, obs_key)
            log.info(f"[{n_domains=}] JSD: {jsd}, FIDE: {fide}, SVG: {svg}")

    if config.post_training.log_umap:
        _log_umap(model, config)


def _log_umap(model: novae.Novae, config: Config, n_obs_th: int = 500_000):
    for adata in model.adatas:
        if "novae_tissue" in adata.uns:
            adata.obs["tissue"] = adata.uns["novae_tissue"]

    for n_domains in config.post_training.n_domains:
        obs_key = model.assign_domains(n_domains=n_domains)
        model.batch_effect_correction(obs_key=obs_key)

        latent_conc = np.concat([adata.obsm[Keys.REPR_CORRECTED] for adata in model.adatas], axis=0)
        obs_conc = pd.concat([adata.obs for adata in model.adatas], axis=0)
        adata_conc = AnnData(obsm={Keys.REPR_CORRECTED: latent_conc}, obs=obs_conc)

        if adata_conc.n_obs > n_obs_th:
            sc.pp.subsample(adata_conc, n_obs=n_obs_th)

        sc.pp.neighbors(adata_conc, use_rep=Keys.REPR_CORRECTED)
        sc.tl.umap(adata_conc)

        colors = [obs_key]
        for key in ["tissue", "technology"]:
            if key in adata_conc.obs:
                colors.append(key)

        sc.pl.umap(adata_conc, color=colors, show=False)
        log_plt_figure(f"umap_{n_domains=}")


def _save_h5ad(model: novae.Novae, config: Config):
    for n_domains in config.post_training.n_domains:
        model.assign_domains(n_domains=n_domains)

    res_dir = novae.utils.repository_root() / "data" / "results" / wandb.run.name
    res_dir.mkdir(parents=True, exist_ok=True)

    for adata in model.adatas:
        out_path = res_dir / f"{id(adata)}.h5ad"
        log.info(f"Writing adata file to {out_path}: {adata}")
        adata.write_h5ad(out_path)