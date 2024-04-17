import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from anndata import AnnData
from umap import UMAP

from .._constants import QC, REPR


def partial_umap(adata: AnnData, obsm: str, n_obs: int = 100_000, min_dist: float = 0.5, n_neighbors: int = 15) -> UMAP:
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


def latent_plot(
    model, adata: AnnData, obsm=REPR, n_obs: int = 100_000, min_dist: float = 0.5, n_neighbors: int = 15, **kwargs
):
    adata.obsm["X_umap"] = np.zeros((adata.n_obs, 2))

    if n_obs < adata.n_obs:
        indices = np.random.choice(adata.n_obs, size=n_obs, replace=False)
    else:
        indices = np.arange(adata.n_obs)

    X = adata[indices].obsm[obsm]
    C = model.swav_head.prototypes.data.T.numpy(force=True)
    X_conc = np.concatenate([X, C], axis=0)

    reducer = UMAP(min_dist=min_dist, n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(X_conc)

    adata.obsm["X_umap"][indices] = embedding[: len(X)]

    C_umap = embedding[-len(C) :]

    plot_partial_umap(adata, **kwargs)
    plt.scatter(C_umap[:, 0], C_umap[:, 1], color="k", s=1)
