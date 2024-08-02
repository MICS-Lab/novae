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
from ..plot import plot_latent
from .eval import heuristic, mean_fide_score

DEFAULT_N_DOMAINS = [7]


class LogProtoCovCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, model: Novae) -> None:
        C = model.swav_head.prototypes.data.numpy(force=True)

        plt.figure(figsize=(10, 10))
        sns.clustermap(np.cov(C))
        wandb.log({"prototypes_covariance": wandb.Image(plt)})
        plt.close()


class LogTissuePrototypeWeights(Callback):
    def on_train_epoch_end(self, trainer: Trainer, model: Novae) -> None:
        if model.swav_head.queue is None:
            return

        tissue_prototype_weights = (
            model.swav_head.sinkhorn(model.swav_head.queue.mean(dim=1)).numpy(force=True)
            * model.swav_head.num_prototypes
        )

        plt.figure(figsize=(10, 10))
        sns.clustermap(
            tissue_prototype_weights, yticklabels=list(model.swav_head.tissue_label_encoder.keys()), vmin=0.8, vmax=1.2
        )
        wandb.log({"tissue_prototype_weights": wandb.Image(plt)})
        plt.close()

        tissue_prototype_weights[tissue_prototype_weights < 1] = 0
        plt.figure(figsize=(10, 10))
        sns.clustermap(
            tissue_prototype_weights, yticklabels=list(model.swav_head.tissue_label_encoder.keys()), vmin=0.9, vmax=1.2
        )
        wandb.log({"tissue_prototype_weights_bin": wandb.Image(plt)})
        plt.close()


class ValidationCallback(Callback):
    def __init__(
        self,
        adatas: list[AnnData] | None,
        accelerator: str = "cpu",
        num_workers: int = 0,
        slide_name_key: str = "slide_id",
    ):
        assert adatas is None or len(adatas) == 1, "ValidationCallback only supports single slide mode for now"
        self.adata = adatas[0] if adatas is not None else None
        self.accelerator = accelerator
        self.num_workers = num_workers
        self.slide_name_key = slide_name_key

        self._max_heuristic = 0.0

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        if self.adata is None:
            return

        model.mode.trained = True  # trick to avoid assert error in compute_representation

        model.compute_representation(
            self.adata, accelerator=self.accelerator, num_workers=self.num_workers, zero_shot=True
        )
        model.swav_head.hierarchical_clustering()

        n_domain = DEFAULT_N_DOMAINS[0]
        k = n_domain

        obs_key = model.assign_domains(self.adata, k)
        while len(self.adata.obs[obs_key].dropna().unique()) < n_domain:
            k += 1
            obs_key = model.assign_domains(self.adata, k)

        plt.figure()
        sc.pl.spatial(self.adata, color=obs_key, spot_size=20, img_key=None, show=False)
        slide_name_key = self.slide_name_key if self.slide_name_key in self.adata.obs else Keys.SLIDE_ID
        wandb.log({f"val_{n_domain}_{self.adata.obs[slide_name_key].iloc[0]}": wandb.Image(plt)})
        plt.close()

        fide = mean_fide_score(self.adata, obs_key=obs_key, n_classes=n_domain)
        model.log("metrics/val_mean_fide_score", fide)

        heuristic_ = heuristic(self.adata, obs_key=obs_key, n_classes=n_domain)
        model.log("metrics/val_heuristic", heuristic_)

        self._max_heuristic = max(self._max_heuristic, heuristic_)
        model.log("metrics/val_max_heuristic", self._max_heuristic)

        model.mode.zero_shot = False


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
