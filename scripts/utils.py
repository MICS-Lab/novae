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
from novae._constants import Keys, Nums
from novae.monitor import jensen_shannon_divergence, mean_fide_score, mean_normalized_entropy
from novae.monitor.callback import (
    LogProtoCovCallback,
    LogTissuePrototypeWeights,
    PrototypeUMAPCallback,
    ValidationCallback,
)
from novae.monitor.log import log_plt_figure, wandb_results_dir

from .config import Config


def init_wandb_logger(config: Config) -> WandbLogger:
    wandb.init(project=config.project, **config.wandb_init_kwargs)

    if config.sweep:
        sweep_config = dict(wandb.config)

        fit_kwargs_args = ["lr", "patience", "min_delta"]

        for arg in fit_kwargs_args:
            if arg in sweep_config:
                config.fit_kwargs[arg] = sweep_config[arg]
                del sweep_config[arg]
        for key, value in sweep_config.items():
            if hasattr(Nums, key):
                log.info(f"Nums.{key} = {value}")
                setattr(Nums, key, value)
            else:
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

    validation_callback = [ValidationCallback(adatas_val, **_get_hardware_kwargs(config))] if adatas_val else []

    if config.sweep:
        return validation_callback

    return [
        *validation_callback,
        ModelCheckpoint(monitor="metrics/val_heuristic", mode="max", save_last=True, save_top_k=1),
        LogProtoCovCallback(),
        LogTissuePrototypeWeights(),
        PrototypeUMAPCallback(),
    ]


def read_config(args: argparse.Namespace) -> Config:
    with open(novae.utils.repository_root() / "scripts" / "config" / args.config) as f:
        config = yaml.safe_load(f)
        config = Config(**config, sweep=args.sweep)

        log.info(f"Using {config.seed}")
        L.seed_everything(config.seed)

        return config


def post_training(model: novae.Novae, adatas: list[AnnData], config: Config):  # noqa: C901
    wandb.log({"num_parameters": sum(p.numel() for p in model.parameters())})

    keys_repr = ["log_umap", "log_metrics", "log_domains"]
    if any(getattr(config.post_training, key) for key in keys_repr):
        model.compute_representations(adatas, **_get_hardware_kwargs(config), zero_shot=config.zero_shot)
        for n_domains in config.post_training.n_domains:
            try:
                obs_key = model.assign_domains(adatas, n_domains=n_domains)
            except:
                log.warning(f"Assigning domains with level as n_domains={n_domains} failed")
                obs_key = model.assign_domains(adatas, level=n_domains)

    if config.post_training.log_domains:
        for n_domains in config.post_training.n_domains:
            try:
                obs_key = model.assign_domains(adatas, n_domains=n_domains)
            except:
                log.warning(f"Assigning domains with level as n_domains={n_domains} failed")
                obs_key = model.assign_domains(adatas, level=n_domains)
            novae.plot.domains(adatas, obs_key, show=False)
            log_plt_figure(f"domains_{n_domains=}")

    if config.post_training.log_metrics:
        for n_domains in config.post_training.n_domains:
            try:
                obs_key = model.assign_domains(adatas, n_domains=n_domains)
            except:
                log.warning(f"Assigning domains with level as n_domains={n_domains} failed")
                obs_key = model.assign_domains(adatas, level=n_domains)
            jsd = jensen_shannon_divergence(adatas, obs_key)
            fide = mean_fide_score(adatas, obs_key, n_classes=n_domains)
            mne = mean_normalized_entropy(adatas, n_classes=n_domains, obs_key=obs_key)
            log.info(f"[{n_domains=}] JSD: {jsd}, FIDE: {fide}, MNE: {mne}")
            wandb.log({
                f"metrics/jsd_{n_domains}_domains": jsd,
                f"metrics/fid_{n_domains}_domainse": fide,
                f"metrics/mne_{n_domains}_domains": mne,
                f"metrics/train_heuristic_{n_domains}_domains": fide * mne,
            })

    if config.post_training.log_umap:
        _log_umap(model, adatas, config)

    if config.post_training.save_h5ad:
        for adata in adatas:
            if config.post_training.delete_X:
                del adata.X
                if "counts" in adata.layers:
                    del adata.layers["counts"]
            _save_h5ad(adata)


def _log_umap(model: novae.Novae, adatas: list[AnnData], config: Config, n_obs_th: int = 500_000):
    for adata in adatas:
        if "novae_tissue" in adata.uns:
            adata.obs["tissue"] = adata.uns["novae_tissue"]

    for n_domains in config.post_training.n_domains:
        try:
            obs_key = model.assign_domains(adatas, n_domains=n_domains)
        except:
            log.warning(f"Assigning domains with level as n_domains={n_domains} failed")
            obs_key = model.assign_domains(adatas, level=n_domains)
        model.batch_effect_correction(adatas, obs_key=obs_key)

        latent_conc = np.concatenate([adata.obsm[Keys.REPR_CORRECTED] for adata in adatas], axis=0)
        obs_conc = pd.concat([adata.obs for adata in adatas], axis=0, join="inner")
        adata_conc = AnnData(obsm={Keys.REPR_CORRECTED: latent_conc}, obs=obs_conc)

        if "cell_id" in adata_conc.obs:
            del adata_conc.obs["cell_id"]  # can't be saved for some reasons
        _save_h5ad(adata_conc, "adata_conc")

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


def _save_h5ad(adata: AnnData, stem: str | None = None):
    if stem is None:
        stem = adata.obs["slide_id"].iloc[0] if "slide_id" in adata.obs else str(id(adata))

    out_path = wandb_results_dir() / f"{stem}.h5ad"
    log.info(f"Writing adata file to {out_path}: {adata}")
    adata.write_h5ad(out_path)
