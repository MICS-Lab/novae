from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from lightning import Trainer
from lightning.pytorch.callbacks import Callback

import wandb

from .._constants import Keys
from ..model import Novae
from ..utils._plot import plot_latent
from .eval import jensen_shannon_divergence, mean_fide_score, mean_svg_score

DEFAULT_N_DOMAINS = [7, 14]


class ComputeSwavOutputsCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, model: Novae) -> None:
        model._trained = True  # trick to assert error in compute_representation
        model.compute_representation()
        model.swav_head.hierarchical_clustering()

        for adata in model.adatas:
            obs_key = model.assign_domains(adata, DEFAULT_N_DOMAINS[0])

        model.batch_effect_correction(None, obs_key)


class LogDomainsCallback(Callback):
    def __init__(self, slide_name_key: str = "slide_id", **plot_kwargs) -> None:
        super().__init__()
        self.slide_name_key = slide_name_key
        self.plot_kwargs = plot_kwargs

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        for adata in model.adatas:
            self.log_domains_plots(model, adata, **self.plot_kwargs)

    def log_domains_plots(self, model: Novae, adata: AnnData | list[AnnData], n_domains: list = DEFAULT_N_DOMAINS):
        for k in n_domains:
            obs_key = model.assign_domains(adata, k)
            sc.pl.spatial(adata, color=obs_key, spot_size=20, img_key=None, show=False)
            slide_name_key = self.slide_name_key if self.slide_name_key in adata.obs else Keys.SLIDE_ID
            wandb.log({f"{obs_key}_{adata.obs[slide_name_key].iloc[0]}": wandb.Image(plt)})


class LogProtoCovCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, model: Novae) -> None:
        C = model.swav_head.prototypes.data.numpy(force=True)
        sns.clustermap(np.cov(C))
        wandb.log({"prototypes_covariance": wandb.Image(plt)})


class ValidationCallback(Callback):
    def __init__(self, adatas: list[AnnData] | None, accelerator: str | None = None, num_workers: int | None = None):
        self.adatas = adatas
        self.accelerator = accelerator
        self.num_workers = num_workers

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        if self.adatas is None:
            return

        model._trained = True  # trick to assert error in compute_representation
        model.compute_representation(self.adatas, accelerator=self.accelerator, num_workers=self.num_workers)
        model.swav_head.hierarchical_clustering()

        n_classes = DEFAULT_N_DOMAINS[0]

        for adata in self.adatas:
            obs_key = model.assign_domains(adata, n_classes)
            sc.pl.spatial(adata, color=obs_key, spot_size=20, img_key=None, show=False)
            wandb.log({f"val_{obs_key}_{adata.obs[Keys.SLIDE_ID].iloc[0]}": wandb.Image(plt)})

        fide = mean_fide_score(self.adatas, obs_key=obs_key, n_classes=n_classes)
        model.log("metrics/val_mean_fide_score", fide)


class EvalCallback(Callback):
    def __init__(self, n_domains=DEFAULT_N_DOMAINS) -> None:
        super().__init__()
        self.n_domains = n_domains

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        for k in self.n_domains:
            obs_key = f"{Keys.NICHE_PREFIX}{k}"

            fide = mean_fide_score(model.adatas, obs_key=obs_key, n_classes=k)
            jsd = jensen_shannon_divergence(model.adatas, obs_key, model.slide_key)
            svg = mean_svg_score(model.adatas, obsm_key=Keys.REPR, obs_key=obs_key)

            wandb.log({f"metrics/mean_fide_score_{k}": fide})
            wandb.log({f"metrics/jensen_shannon_divergence_{k}": jsd})
            wandb.log({f"metrics/mean_svg_score_{k}": svg})


class LogLatent(Callback):
    def __init__(self, **plot_kwargs) -> None:
        super().__init__()
        self.plot_kwargs = plot_kwargs

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        colors = [f"{Keys.NICHE_PREFIX}{k}" for k in DEFAULT_N_DOMAINS] + [Keys.SLIDE_ID]
        plot_latent(model.adatas, colors, **self.plot_kwargs)
        wandb.log({"latent": wandb.Image(plt)})

        plot_latent(model.adatas, colors, obsm=Keys.REPR_CORRECTED, **self.plot_kwargs)
        wandb.log({"latent_corrected": wandb.Image(plt)})
