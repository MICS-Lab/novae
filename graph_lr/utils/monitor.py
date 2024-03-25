import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

import wandb

from .._constants import SWAV_CLASSES
from ..model import GraphLR


def log_domains_plots(
    model: GraphLR, adata: AnnData | list[AnnData], n_domains: list = [7, 11, 15], suffix=""
):
    if isinstance(adata, list):
        for i, adata_ in adata:
            log_domains_plots(model, adata_, n_domains=n_domains, suffix=f"_sample_{i}")

    for k in n_domains:
        obs_key = f"{SWAV_CLASSES}_{k}"
        adata.obs[obs_key] = model.swav_head.assign_classes_level(adata.obs[SWAV_CLASSES], k)
        sc.pl.spatial(adata, color=obs_key, spot_size=20, img_key=None, show=False)
        wandb.log({f"{obs_key}{suffix}": wandb.Image(plt)})
