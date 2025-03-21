"""
Copied and updated from squidpy (to not have squidpy as a dependency)
Functions for building graphs from spatial coordinates.
"""

import logging
import warnings
from enum import Enum
from functools import partial
from itertools import chain
from typing import Iterable, Literal, get_args

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.utils import make_index_unique
from scipy.sparse import SparseEfficiencyWarning, block_diag, csr_matrix, spmatrix
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

from .._constants import Keys, Nums
from ._utils import unique_obs

log = logging.getLogger(__name__)
__all__ = ["spatial_neighbors"]

SpatialTechnology = Literal["cosmx", "merscope", "xenium", "visium", "visium_hd"]


class CoordType(Enum):
    GRID = "grid"
    GENERIC = "generic"


def spatial_neighbors(
    adata: AnnData | list[AnnData],
    slide_key: str | None = None,
    radius: tuple[float, float] | float | None = None,
    pixel_size: float | None = None,
    technology: str | SpatialTechnology | None = None,
    coord_type: str | CoordType | None = None,
    n_neighs: int | None = None,
    delaunay: bool | None = None,
    n_rings: int = 1,
    percentile: float | None = None,
    set_diag: bool = False,
    reset_slide_ids: bool = True,
):
    """Create a Delaunay graph from the spatial coordinates of the cells.
    The graph is stored in `adata.obsp['spatial_connectivities']` and `adata.obsp['spatial_distances']`. The long edges
    are removed from the graph according to the `radius` argument (if provided).

    Info:
        This function was updated from [squidpy](https://squidpy.readthedocs.io/en/latest/api/squidpy.gr.spatial_neighbors.html#squidpy.gr.spatial_neighbors).

    Args:
        adata: An `AnnData` object, or a list of `AnnData` objects.
        slide_key: Optional key in `adata.obs` indicating the slide ID of each cell. If provided, the graph is computed for each slide separately.
        radius: `tuple` that prunes the final graph to only contain edges in interval `[min(radius), max(radius)]`. If `float`, uses `[0, radius]`. If `None`, all edges are kept.
        technology: Technology or machine used to generate the spatial data. One of `"cosmx", "merscope", "xenium", "visium", "visium_hd"`. If `None`, uses `adata.obsm["spatial"]`.
        coord_type: Either `"grid"` or `"generic"`. If `"grid"`, the graph is built on a grid. If `"generic"`, the graph is built using the coordinates as they are. By default, uses `"grid"` for Visium/VisiumHD and `"generic"` for other technologies.
        n_neighs: Number of neighbors to consider. If `None`, uses `6` for Visium, `4` for Visium HD, and `None` for generic graphs.
        delaunay: Whether to use Delaunay triangulation to build the graph. If `None`, uses `False` for grid-based graphs and `True` for generic graphs.
        n_rings: See `squidpy.gr.spatial_neighbors` documentation.
        percentile: See `squidpy.gr.spatial_neighbors` documentation.
        set_diag: See `squidpy.gr.spatial_neighbors` documentation.
        reset_slide_ids: Whether to reset the novae slide ids.
    """
    if reset_slide_ids:
        _set_unique_slide_ids(adata, slide_key=slide_key)

    if isinstance(adata, list):
        for adata_ in adata:
            spatial_neighbors(
                adata_,
                slide_key=slide_key,
                radius=radius,
                pixel_size=pixel_size,
                technology=technology,
                coord_type=coord_type,
                n_neighs=n_neighs,
                delaunay=delaunay,
                n_rings=n_rings,
                percentile=percentile,
                set_diag=set_diag,
                reset_slide_ids=False,
            )
        return

    if isinstance(radius, float) or isinstance(radius, int):
        radius = [0.0, float(radius)]

    assert radius is None or len(radius) == 2, "Radius is expected to be a tuple (min_radius, max_radius)"

    assert pixel_size is None or technology is None, "You must choose argument between `pixel_size` and `technology`"

    if technology == "visium":
        n_neighs = 6 if n_neighs is None else n_neighs
        coord_type, delaunay = CoordType.GRID, False
    elif technology == "visium_hd":
        n_neighs = 8 if n_neighs is None else n_neighs
        coord_type, delaunay = CoordType.GRID, False
    elif technology is not None:
        adata.obsm["spatial"] = _technology_coords(adata, technology)

    assert (
        "spatial" in adata.obsm
    ), "Key 'spatial' not found in adata.obsm. This should contain the 2D spatial coordinates of the cells"

    coord_type = CoordType(coord_type or "generic")
    delaunay = True if delaunay is None else delaunay
    n_neighs = 6 if (n_neighs is None and not delaunay) else n_neighs

    log.info(
        f"Computing graph on {adata.n_obs:,} cells (coord_type={coord_type.value}, {delaunay=}, {radius=}, {n_neighs=})"
    )

    slides = adata.obs[Keys.SLIDE_ID].cat.categories
    make_index_unique(adata.obs_names)

    _build_fun = partial(
        _spatial_neighbor,
        coord_type=coord_type,
        n_neighs=n_neighs,
        radius=radius,
        delaunay=delaunay,
        n_rings=n_rings,
        set_diag=set_diag,
        percentile=percentile,
    )

    if len(slides) > 1:
        mats: list[tuple[spmatrix, spmatrix]] = []
        ixs = []  # type: ignore[var-annotated]
        for slide in slides:
            ixs.extend(np.where(adata.obs[Keys.SLIDE_ID] == slide)[0])
            mats.append(_build_fun(adata[adata.obs[Keys.SLIDE_ID] == slide]))
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
        "params": {"radius": radius, "set_diag": set_diag, "n_neighbors": n_neighs, "coord_type": coord_type.value},
    }

    _sanity_check_spatial_neighbors(adata)


def _spatial_neighbor(
    adata: AnnData,
    spatial_key: str = "spatial",
    coord_type: str | CoordType | None = None,
    n_neighs: int = 6,
    radius: float | tuple[float, float] | None = None,
    delaunay: bool = False,
    n_rings: int = 1,
    set_diag: bool = False,
    percentile: float | None = None,
) -> tuple[csr_matrix, csr_matrix]:
    coords = adata.obsm[spatial_key]
    assert coords.shape[1] == 2, "Spatial coordinates must be 2D."

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        if coord_type == CoordType.GRID:
            Adj, Dst = _build_grid(coords, n_neighs=n_neighs, n_rings=n_rings, delaunay=delaunay, set_diag=set_diag)
        elif coord_type == CoordType.GENERIC:
            Adj, Dst = _build_connectivity(
                coords, n_neighs=n_neighs, radius=radius, delaunay=delaunay, return_distance=True, set_diag=set_diag
            )
        else:
            raise NotImplementedError(f"Coordinate type `{coord_type}` is not yet implemented.")

    if coord_type == CoordType.GENERIC and isinstance(radius, Iterable):
        minn, maxx = sorted(radius)[:2]
        mask = (Dst.data < minn) | (Dst.data > maxx)
        a_diag = Adj.diagonal()

        Dst.data[mask] = 0.0
        Adj.data[mask] = 0.0
        Adj.setdiag(a_diag)

    if percentile is not None and coord_type == CoordType.GENERIC:
        threshold = np.percentile(Dst.data, percentile)
        Adj[Dst > threshold] = 0.0
        Dst[Dst > threshold] = 0.0

    Adj.eliminate_zeros()
    Dst.eliminate_zeros()

    return Adj, Dst


def _build_grid(
    coords: np.ndarray, n_neighs: int, n_rings: int, delaunay: bool = False, set_diag: bool = False
) -> tuple[csr_matrix, csr_matrix]:
    if n_rings > 1:
        Adj: csr_matrix = _build_connectivity(
            coords,
            n_neighs=n_neighs,
            neigh_correct=True,
            set_diag=True,
            delaunay=delaunay,
            return_distance=False,
        )
        Res, Walk = Adj, Adj
        for i in range(n_rings - 1):
            Walk = Walk @ Adj
            Walk[Res.nonzero()] = 0.0
            Walk.eliminate_zeros()
            Walk.data[:] = i + 2.0
            Res = Res + Walk
        Adj = Res
        Adj.setdiag(float(set_diag))
        Adj.eliminate_zeros()

        Dst = Adj.copy()
        Adj.data[:] = 1.0
    else:
        Adj, Dst = _build_connectivity(
            coords, n_neighs=n_neighs, neigh_correct=True, delaunay=delaunay, set_diag=set_diag, return_distance=True
        )

    Dst.setdiag(0.0)

    return Adj, Dst


def _build_connectivity(
    coords: np.ndarray,
    n_neighs: int,
    radius: float | tuple[float, float] | None = None,
    delaunay: bool = False,
    neigh_correct: bool = False,
    set_diag: bool = False,
    return_distance: bool = False,
) -> csr_matrix | tuple[csr_matrix, csr_matrix]:
    N = coords.shape[0]
    if delaunay:
        tri = Delaunay(coords)
        indptr, indices = tri.vertex_neighbor_vertices
        Adj = csr_matrix((np.ones_like(indices, dtype=np.float64), indices, indptr), shape=(N, N))

        if return_distance:
            # fmt: off
            dists = np.array(list(chain(*(
                euclidean_distances(coords[indices[indptr[i] : indptr[i + 1]], :], coords[np.newaxis, i, :])
                for i in range(N)
                if len(indices[indptr[i] : indptr[i + 1]])
            )))).squeeze()
            Dst = csr_matrix((dists, indices, indptr), shape=(N, N))
            # fmt: on
    else:
        r = 1 if radius is None else radius if isinstance(radius, (int, float)) else max(radius)
        tree = NearestNeighbors(n_neighbors=n_neighs, radius=r, metric="euclidean")
        tree.fit(coords)

        if radius is None:
            dists, col_indices = tree.kneighbors()
            dists, col_indices = dists.reshape(-1), col_indices.reshape(-1)
            row_indices = np.repeat(np.arange(N), n_neighs)
            if neigh_correct:
                dist_cutoff = np.median(dists) * 1.3  # there's a small amount of sway
                mask = dists < dist_cutoff
                row_indices, col_indices, dists = row_indices[mask], col_indices[mask], dists[mask]
        else:
            dists, col_indices = tree.radius_neighbors()
            row_indices = np.repeat(np.arange(N), [len(x) for x in col_indices])
            dists = np.concatenate(dists)
            col_indices = np.concatenate(col_indices)

        Adj = csr_matrix((np.ones_like(row_indices, dtype=np.float64), (row_indices, col_indices)), shape=(N, N))
        if return_distance:
            Dst = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

    # radius-based filtering needs same indices/indptr: do not remove 0s
    Adj.setdiag(1.0 if set_diag else Adj.diagonal())
    if return_distance:
        Dst.setdiag(0.0)
        return Adj, Dst

    return Adj


def _technology_coords(adata: AnnData, technology: str) -> np.ndarray:
    VALID_TECHNOLOGIES = list(get_args(SpatialTechnology))
    factor: float = 1.0

    assert technology in VALID_TECHNOLOGIES, f"Invalid `technology` argument. Choose one of {VALID_TECHNOLOGIES}"

    assert (
        "spatial" not in adata.obsm
    ), "Running `novae.spatial_neighbors` with `technology` but `adata.obsm['spatial']` already exists."

    if technology == "cosmx":
        columns = ["CenterX_global_px", "CenterY_global_px"]
        factor = 0.120280945

    if technology == "merscope":
        columns = ["center_x", "center_y"]

    if technology == "xenium":
        columns = ["x_centroid", "y_centroid"]

    assert all(column in adata.obs for column in columns), f"For {technology=}, you must have {columns} in `adata.obs`"

    return adata.obs[columns].values * factor


def _set_unique_slide_ids(adatas: AnnData | list[AnnData], slide_key: str | None) -> None:
    adatas = [adatas] if isinstance(adatas, AnnData) else adatas

    # we will re-create the slide-ids, even if already set
    for adata in adatas:
        if Keys.SLIDE_ID in adata.obs:
            del adata.obs[Keys.SLIDE_ID]

    if slide_key is None:  # each adata has its own slide ID
        for adata in adatas:
            adata.obs[Keys.SLIDE_ID] = pd.Series(id(adata), index=adata.obs_names, dtype="category")
        return

    assert all(slide_key in adata.obs for adata in adatas), f"{slide_key=} must be in all `adata.obs`"

    slides_ids = [unique_obs(adata, slide_key) for adata in adatas]

    if len(set.union(*slides_ids)) == sum(len(slide_ids) for slide_ids in slides_ids):
        for adata in adatas:
            adata.obs[Keys.SLIDE_ID] = adata.obs[slide_key].astype("category")
        return

    log.warning("Some slides may have the same `slide_key` values. We add `id(adata)` id to the slide IDs.")

    for adata in adatas:
        values: pd.Series = f"{id(adata)}_" + adata.obs[slide_key].astype(str)
        adata.obs[Keys.SLIDE_ID] = values.astype("category")


def _sanity_check_spatial_neighbors(adata: AnnData):
    assert adata.obsp[Keys.ADJ].getnnz() > 0, "No neighbors found. Please check your `radius` parameter."

    mean_distance = adata.obsp[Keys.ADJ].data.mean()
    max_distance = adata.obsp[Keys.ADJ].data.max()

    if max_distance / mean_distance > Nums.MAX_MEAN_DISTANCE_RATIO:
        log.warning(
            f"The maximum distance between neighbors is {max_distance:.1f}, which is very high compared "
            f"to the mean distance of {mean_distance:.1f}.\n Consider re-running `novae.spatial_neighbors` with a different `radius` threshold."
        )

    mean_ngh = adata.obsp[Keys.ADJ].getnnz(axis=1).mean()

    if mean_ngh <= Nums.MEAN_NGH_TH_WARNING:
        log.warning(
            f"The mean number of neighbors is {mean_ngh}, which is very low. Consider re-running `novae.spatial_neighbors` with a different `radius` threshold."
        )
