from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from lightning import Trainer
from lightning.pytorch.callbacks import Callback

import wandb

from .._constants import CODES, REPR, SLIDE_KEY, SWAV_CLASSES
from ..model import Novae
from ..utils._plot import latent_plot
from .eval import mean_pide_score

DEFAULT_N_DOMAINS = [7, 14]


class ComputeSwavOutputsCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, model: Novae) -> None:
        model.codes()
        model.swav_classes()
        model.swav_head.hierarchical_clustering()


class LogDomainsCallback(Callback):
    def __init__(self, **plot_kwargs) -> None:
        super().__init__()
        self.plot_kwargs = plot_kwargs

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        for adata in model.adatas:
            self.log_domains_plots(model, adata, **self.plot_kwargs)

    def log_domains_plots(self, model: Novae, adata: AnnData | list[AnnData], n_domains: list = DEFAULT_N_DOMAINS):
        for k in n_domains:
            obs_key = model.assign_domains(adata, k)
            sc.pl.spatial(adata, color=obs_key, spot_size=20, img_key=None, show=False)
            wandb.log({f"{obs_key}_{adata.obs[SLIDE_KEY].iloc[0]}": wandb.Image(plt)})


class EvalCallback(Callback):
    def __init__(self, n_domains=DEFAULT_N_DOMAINS) -> None:
        super().__init__()
        self.n_domains = n_domains

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        for k in self.n_domains:
            obs_key = f"{SWAV_CLASSES}_{k}"
            wandb.log({f"metrics/mean_pide_score_{k}": mean_pide_score(model.adatas, obs_key=obs_key)})


class LogLatent(Callback):
    def __init__(self, **plot_kwargs) -> None:
        super().__init__()
        self.plot_kwargs = plot_kwargs

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        colors = [f"{SWAV_CLASSES}_{k}" for k in DEFAULT_N_DOMAINS] + [SLIDE_KEY]
        obs = pd.concat([adata.obs[colors] for adata in model.adatas], axis=0)
        obs.reset_index()
        adata = AnnData(obs=obs)

        representation = np.concatenate([adata.obsm[REPR] for adata in model.adatas])
        adata.obsm[REPR] = representation

        latent_plot(model, adata, color=colors, **self.plot_kwargs)
        wandb.log({"latent": wandb.Image(plt)})
