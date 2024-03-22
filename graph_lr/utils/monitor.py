import io
from typing import Callable

import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
from PIL import Image

import wandb

from .._constants import SWAV_CLASSES
from ..model import GraphLR


def wandb_plt_image(fun: Callable, figsize: tuple[int, int] = [7, 5]):
    """Transform a matplotlib figure into a wandb Image

    Args:
        fun: Function that makes the plot - do not plt.show().
        figsize: Matplotlib figure size.

    Returns:
        The wandb Image to be logged.
    """
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    fun()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    return wandb.Image(Image.open(img_buf))


def show_niches(model: GraphLR, adata: AnnData, n_domains: list = [7, 11, 15]):
    for k in n_domains:
        obs_key = f"{SWAV_CLASSES}_{k}"
        adata.obs[obs_key] = model.swav_head.assign_classes_level(adata.obs[SWAV_CLASSES], k)
        sc.pl.spatial(adata, color=obs_key, spot_size=20, img_key=None, show=False)
        wandb.log({obs_key: plt})
