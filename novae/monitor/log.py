from __future__ import annotations

import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

import wandb

from .._constants import SWAV_CLASSES
from ..model import Novae
from .eval import mean_pide_score


def log_domains_plots(model: Novae, adata: AnnData | list[AnnData], n_domains: list = [7, 11, 15], suffix=""):
    if isinstance(adata, list):
        for i, adata_ in enumerate(adata):
            log_domains_plots(model, adata_, n_domains=n_domains, suffix=f"_sample_{i}")
        return

    for k in n_domains:
        obs_key = model.assign_domains(adata, k)
        sc.pl.spatial(adata, color=obs_key, spot_size=20, img_key=None, show=False)
        wandb.log({f"{obs_key}{suffix}": wandb.Image(plt)})


def log_metrics(adata: AnnData | list[AnnData], n_domains: list = [7, 11, 15]):
    for k in n_domains:
        obs_key = f"{SWAV_CLASSES}_{k}"
        wandb.log({f"metrics/mean_pide_score_{k}": mean_pide_score(adata, obs_key=obs_key)})
