"""
Copied and updated from squidpy (to not have squidpy as a dependency)
Functions for building graphs from spatial coordinates.
"""

from __future__ import annotations

import logging
import warnings
from functools import partial
from itertools import chain
from typing import Iterable

import numpy as np
from anndata import AnnData
from anndata.utils import make_index_unique
from scipy.sparse import SparseEfficiencyWarning, block_diag, csr_matrix, spmatrix
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import euclidean_distances

log = logging.getLogger(__name__)
__all__ = ["spatial_neighbors"]


def spatial_neighbors(
    adata: AnnData,
    radius: tuple[float, float] | float | None = 100,
    slide_key: str | None = None,
    pixel_size: float | None = None,
    technology: str | None = None,
    percentile: float | None = None,
    set_diag: bool = False,
):
    """Create a Delaunay graph from the spatial coordinates of the cells (in microns).
    The graph is stored in `adata.obsp['spatial_connectivities']` and `adata.obsp['spatial_distances']`. The long edges
    are removed from the graph according to the `radius` argument (if provided).

    Note:
        The spatial coordinates are expected to be in microns, and stored in `adata.obsm["spatial"]`.
        If the coordinates are in pixels, set `pixel_size` to the size of a pixel in microns.
        If you don't know the `pixel_size`, or if you don't have `adata.obsm["spatial"]`, you can also provide the `technology` argument.

    Info:
        This function was updated from [squidpy](https://squidpy.readthedocs.io/en/latest/api/squidpy.gr.spatial_neighbors.html#squidpy.gr.spatial_neighbors).

    Args:
        adata: AnnData object
        radius: `tuple` that prunes the final graph to only contain edges in interval `[min(radius), max(radius)]` microns. If `float`, uses `[0, radius]`. If `None`, all edges are kept.
        slide_key: Optional key in `adata.obs` indicating the slide ID of each cell. If provided, the graph is computed for each slide separately.
        pixel_size: Number of microns in one pixel of the image (use this argument if `adata.obsm["spatial"]` is in pixels). If `None`, the coordinates are assumed to be in microns.
        technology: Technology or machine used to generate the spatial data. One of `"cosmx", "merscope", "xenium"`. If `None`, the coordinates are assumed to be in microns.
        percentile: Percentile of the distances to use as threshold.
        set_diag: Whether to set the diagonal of the spatial connectivities to `1.0`.
    """
    if isinstance(radius, float) or isinstance(radius, int):
        radius = [0.0, float(radius)]

    assert radius is None or len(radius) == 2, "Radius is expected to be a tuple (min_radius, max_radius)"

    assert pixel_size is None or technology is None, "You must choose argument between `pixel_size` and `technology`"

    if technology is not None:
        adata.obsm["spatial"] = _technology_coords(adata, technology)

    assert (
        "spatial" in adata.obsm
    ), "Key 'spatial' not found in adata.obsm. This should contain the 2D spatial coordinates of the cells"

    if pixel_size is not None:
        assert (
            "spatial_pixel" not in adata.obsm
        ), "Do nott run `novae.utils.spatial_neighbors` twice ('spatial_pixel' already present in `adata.obsm`)."
        adata.obsm["spatial_pixel"] = adata.obsm["spatial"].copy()
        adata.obsm["spatial"] = adata.obsm["spatial"] * pixel_size

    log.info(f"Computing delaunay graph on {adata.n_obs:,} cells (radius threshold: {radius} microns)")

    if slide_key is not None:
        adata.obs[slide_key] = adata.obs[slide_key].astype("category")
        slides = adata.obs[slide_key].cat.categories
        make_index_unique(adata.obs_names)
    else:
        slides = [None]

    _build_fun = partial(
        _spatial_neighbor,
        set_diag=set_diag,
        radius=radius,
        percentile=percentile,
    )

    if slide_key is not None:
        mats: list[tuple[spmatrix, spmatrix]] = []
        ixs = []  # type: ignore[var-annotated]
        for slide in slides:
            ixs.extend(np.where(adata.obs[slide_key] == slide)[0])
            mats.append(_build_fun(adata[adata.obs[slide_key] == slide]))
        ixs = np.argsort(ixs)  # type: ignore[assignment] # invert
        Adj = block_diag([m[0] for m in mats], format="csr")[ixs, :][:, ixs]
        Dst = block_diag([m[1] for m in mats], format="csr")[ixs, :][:, ixs]
    else:
        Adj, Dst = _build_fun(adata)

    adata.obsp["spatial_connectivities"] = Adj
    adata.obsp["spatial_distances"] = Dst

    adata.uns["spatial_neighbors"] = {
        "connectivities_key": "spatial_connectivities",
        "distances_key": "spatial_distances",
        "params": {"radius": radius, "set_diag": set_diag},
    }


def _spatial_neighbor(
    adata: AnnData,
    radius: float | tuple[float, float] | None = None,
    set_diag: bool = False,
    percentile: float | None = None,
) -> tuple[csr_matrix, csr_matrix]:
    coords = adata.obsm["spatial"]

    assert coords.shape[1] == 2, f"adata.obsm['spatial'] has {coords.shape[1]} dimension(s). Expected 2."

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        Adj, Dst = _build_connectivity(
            coords,
            set_diag=set_diag,
        )

    if isinstance(radius, Iterable):
        minn, maxx = sorted(radius)[:2]  # type: ignore[var-annotated]
        mask = (Dst.data < minn) | (Dst.data > maxx)
        a_diag = Adj.diagonal()

        Dst.data[mask] = 0.0
        Adj.data[mask] = 0.0
        Adj.setdiag(a_diag)

    if percentile is not None:
        threshold = np.percentile(Dst.data, percentile)
        Adj[Dst > threshold] = 0.0
        Dst[Dst > threshold] = 0.0

    Adj.eliminate_zeros()
    Dst.eliminate_zeros()

    return Adj, Dst


def _build_connectivity(
    coords: np.ndarray,
    set_diag: bool = False,
) -> csr_matrix | tuple[csr_matrix, csr_matrix]:
    N = coords.shape[0]

    tri = Delaunay(coords)
    indptr, indices = tri.vertex_neighbor_vertices
    Adj = csr_matrix((np.ones_like(indices, dtype=np.float64), indices, indptr), shape=(N, N))

    # fmt: off
    dists = np.array(list(chain(*(
        euclidean_distances(coords[indices[indptr[i] : indptr[i + 1]], :], coords[np.newaxis, i, :])
        for i in range(N)
        if len(indices[indptr[i] : indptr[i + 1]])
    )))).squeeze()
    Dst = csr_matrix((dists, indices, indptr), shape=(N, N))
    # fmt: on

    # radius-based filtering needs same indices/indptr: do not remove 0s
    Adj.setdiag(1.0 if set_diag else Adj.diagonal())
    Dst.setdiag(0.0)

    return Adj, Dst


def _technology_coords(adata: AnnData, technology: str) -> np.ndarray:
    VALID_TECHNOLOGIES = ["cosmx", "merscope", "xenium"]
    factor: float = 1.0

    assert technology in VALID_TECHNOLOGIES, f"Invalid `technology` argument. Choose one of {VALID_TECHNOLOGIES}"

    if technology == "cosmx":
        columns = ["CenterX_global_px", "CenterY_global_px"]
        factor = 0.120280945

    if technology == "merscope":
        columns = ["center_x", "center_y"]

    if technology == "xenium":
        columns = ["x_centroid", "y_centroid"]

    assert all(column in adata.obs for column in columns), f"For {technology=}, you must have {columns} in `adata.obs`"

    return adata.obs[columns].values * factor
