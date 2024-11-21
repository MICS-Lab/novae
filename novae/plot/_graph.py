from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from .. import utils
from .._constants import Keys
from ._utils import get_categorical_color_palette


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


def _domains_hierarchy(
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
            plt.hlines(original_ymax - level, 0, 1e5, colors="r", linestyles="dashed")

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
        plt.annotate(
            f"D{2 * size - max_level + int(y)}",
            (x, y),
            xytext=(0, -8),
            textcoords="offset points",
            va="top",
            ha="center",
        )

    plt.yticks(original_ticks, original_ymax - original_ticks)

    plt.xlabel(None)
    plt.ylabel("Domains level")
    plt.title("Domains hierarchy")
    plt.xlabel("Number of domains in node (or prototype ID if no parenthesis)")
    sns.despine(offset=10, trim=True, bottom=True)


def paga(adata: AnnData, obs_key: str | None = None, **paga_plot_kwargs: int):
    """Plot a PAGA graph.

    Info:
        Currently, this function only supports one slide per call.

    Args:
        adata: An AnnData object.
        obs_key: Name of the key from `adata.obs` containing the Novae domains. By default, the last available domain key is shown.
        **paga_plot_kwargs: Additional arguments for `sc.pl.paga`.
    """
    assert isinstance(adata, AnnData), f"For now, only AnnData objects are supported, received {type(adata)}"

    obs_key = utils.check_available_domains_key([adata], obs_key)

    get_categorical_color_palette([adata], obs_key)

    adata_clean = adata[~adata.obs[obs_key].isna()]

    if "paga" not in adata.uns or adata.uns["paga"]["groups"] != obs_key:
        sc.pp.neighbors(adata_clean, use_rep=Keys.REPR)
        sc.tl.paga(adata_clean, groups=obs_key)

        adata.uns["paga"] = adata_clean.uns["paga"]
        adata.uns[f"{obs_key}_sizes"] = adata_clean.uns[f"{obs_key}_sizes"]

    sc.pl.paga(adata_clean, title=f"PAGA graph ({obs_key})", show=False, **paga_plot_kwargs)
    sns.despine(offset=10, trim=True, bottom=True)


def connectivities(
    adata: AnnData,
    ngh_threshold: int | None = None,
    cell_size: int = 5,
    linewidths: float = 1,
    line_color: str = "#333",
    cmap="rocket",
    color_isolated_cells: str = "orangered",
):
    """Show the graph of the spatial connectivities between cells. By default,
    the cells are colored by the number of neighbors. Use `ngh_threshold` to show
    only the cells with a number of neighbors below this threshold (good for quality control).

    Args:
        adata: An AnnData object.
        ngh_threshold: If not `None`, only cells with a number of neighbors below this threshold are shown (with color `color_isolated_cells`).
        cell_size: Size of the dots for each cell.
        linewidths: Width of the lines/edges connecting the cells.
        line_color: Color of the lines/edges.
        cmap: Name of the colormap to use for the number of neighbors.
        color_isolated_cells: Color for the cells with a number of neighbors below `ngh_threshold` (if not `None`).
    """
    utils.check_has_spatial_adjancency(adata)

    X, A = adata.obsm["spatial"], adata.obsp[Keys.ADJ]

    _, ax = plt.subplots()
    ax.invert_yaxis()
    ax.axes.set_aspect("equal")

    rows, cols = A.nonzero()
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]
    edge_segments = np.stack([X[rows], X[cols]], axis=1)
    edges = LineCollection(edge_segments, color=line_color, linewidths=linewidths, zorder=1)
    ax.add_collection(edges)

    n_neighbors = (A > 0).sum(1).A1

    if ngh_threshold is None:
        _ = plt.scatter(X[:, 0], X[:, 1], c=n_neighbors, s=cell_size, zorder=2, cmap=cmap)
        plt.colorbar(_)
    else:
        isolated_cells = n_neighbors < ngh_threshold
        plt.scatter(X[isolated_cells, 0], X[isolated_cells, 1], color=color_isolated_cells, s=cell_size, zorder=2)
