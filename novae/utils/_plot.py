from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from umap import UMAP

from .._constants import Keys


def _leaves_count(clustering: AgglomerativeClustering) -> np.ndarray:
    counts = np.zeros(clustering.children_.shape[0])
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    return counts


def plot_niches_hierarchy(
    clustering: AgglomerativeClustering,
    max_level: int = 10,
    hline_level: int | list[int] | None = None,
    leaf_font_size: int = 10,
    **kwargs,
) -> None:
    assert max_level > 1

    size = clustering.children_.shape[0]
    original_ymax = max_level + 1
    original_ticks = np.arange(1, original_ymax)
    height = original_ymax + np.arange(size) - size

    if hline_level is not None:
        hline_level = [hline_level] if isinstance(hline_level, int) else hline_level
        for level in hline_level:
            plt.hlines(original_ymax - hline_level, 0, 1e5, colors="r", linestyles="dashed")

    linkage_matrix = np.column_stack([clustering.children_, height.clip(0), _leaves_count(clustering)]).astype(float)

    ddata = dendrogram(
        linkage_matrix,
        color_threshold=-1,
        leaf_font_size=leaf_font_size,
        p=max_level + 1,
        truncate_mode="lastp",
        above_threshold_color="#ccc",
        **kwargs,
    )

    for i, d in zip(ddata["icoord"][::-1], ddata["dcoord"][::-1]):
        x, y = 0.5 * sum(i[1:3]), d[1]
        plt.plot(x, y, "ko")
        plt.annotate(f"N{size - 1 + int(y)}", (x, y), xytext=(0, -8), textcoords="offset points", va="top", ha="center")

    plt.yticks(original_ticks, original_ymax - original_ticks)

    plt.xlabel(None)
    plt.ylabel("Niche level")
    plt.title("Niches hierarchy")
    plt.xlabel("Number of niches in node (or prototype ID if no parenthesis)")
    sns.despine(offset=10, trim=True, bottom=True)


def partial_umap(
    adata: AnnData, obsm: str = Keys.REPR, n_obs: int = 100_000, min_dist: float = 0.5, n_neighbors: int = 30
) -> UMAP:
    adata.obsm["X_umap"] = np.zeros((adata.n_obs, 2))

    if n_obs < adata.n_obs:
        indices = np.random.choice(adata.n_obs, size=n_obs, replace=False)
    else:
        indices = np.arange(adata.n_obs)

    X = adata[indices].obsm[obsm]

    reducer = UMAP(min_dist=min_dist, n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(X)

    adata.obsm["X_umap"][indices] = embedding

    return reducer


def plot_partial_umap(adata: AnnData, **kwargs):
    adata_sub = adata[adata.obsm["X_umap"].sum(1) != 0].copy()
    sc.pl.umap(adata_sub, show=False, **kwargs)


def plot_latent(adatas: list[AnnData], colors, obsm: str = Keys.REPR, **kwargs):
    obs = pd.concat([adata.obs[colors] for adata in adatas], axis=0)
    obs.reset_index()
    adata = AnnData(obs=obs)

    representation = np.concatenate([adata.obsm[obsm] for adata in adatas])
    adata.obsm[obsm] = representation

    partial_umap(adata, obsm=obsm)
    plot_partial_umap(adata, color=colors, **kwargs)
