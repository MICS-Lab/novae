from itertools import chain

import anndata
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
from sklearn.metrics import euclidean_distances

import novae
from novae._constants import Keys
from novae.data.dataset import _to_adjacency_local, _to_adjacency_view
from novae.utils.build import _set_unique_slide_ids

from ._utils import adata, adata_concat, adata_line

true_connectivities = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0],
])


def test_build():
    connectivities = adata.obsp["spatial_connectivities"]

    assert connectivities.shape[0] == adata.n_obs

    assert (connectivities.todense() == true_connectivities).all()


def test_set_unique_slide_ids():
    adatas = novae.data.toy_dataset(
        xmax=200,
        n_panels=2,
        n_slides_per_panel=1,
        n_vars=30,
        slide_ids_unique=False,
    )

    _set_unique_slide_ids(adatas, slide_key="slide_key")

    assert adatas[0].obs[Keys.SLIDE_ID].iloc[0] == f"{id(adatas[0])}_slide_0"

    adatas = novae.data.toy_dataset(
        xmax=200,
        n_panels=2,
        n_slides_per_panel=1,
        n_vars=30,
        slide_ids_unique=True,
    )

    _set_unique_slide_ids(adatas, slide_key="slide_key")

    assert adatas[0].obs[Keys.SLIDE_ID].iloc[0] == "slide_0_0"


def test_build_slide_key():
    adata_ = adata_concat.copy()
    novae.utils.spatial_neighbors(adata_, radius=1.5, slide_key="slide_key")

    true_connectivities = np.array([
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    ])

    assert (adata_.obsp["spatial_connectivities"].todense() == true_connectivities).all()


def test_build_slide_key_disjoint_indices():
    adata = novae.data.toy_dataset(
        n_panels=1,
        n_slides_per_panel=1,
        xmax=100,
        n_domains=2,
    )[0]

    adata2 = adata.copy()
    adata2.obs["slide_key"] = "slide_key2"
    adata2.obs_names = adata2.obs_names + "_2"

    novae.utils.spatial_neighbors(adata)
    novae.utils.spatial_neighbors(adata2)

    n1, n2 = len(adata.obsp["spatial_connectivities"].data), len(adata2.obsp["spatial_connectivities"].data)
    assert n1 == n2

    adata_concat = anndata.concat([adata, adata2], axis=0).copy()

    novae.utils.spatial_neighbors(adata_concat, slide_key="slide_key")

    assert len(adata_concat.obsp["spatial_connectivities"].data) == n1 + n2


def test_build_technology():
    adata_cosmx = adata.copy()
    adata_cosmx.obs[["CenterX_global_px", "CenterY_global_px"]] = adata_cosmx.obsm["spatial"]
    del adata_cosmx.obsm["spatial"]
    novae.utils.spatial_neighbors(adata_cosmx, technology="cosmx")

    del adata_cosmx.obs["CenterY_global_px"]

    # one column is missing in obs
    with pytest.raises(AssertionError):
        novae.utils.spatial_neighbors(adata_cosmx, technology="cosmx")


def test_invalid_build():
    adata_invalid = anndata.AnnData(obs=pd.DataFrame(index=["0", "1", "2"]))

    with pytest.raises(AssertionError):
        novae.utils.spatial_neighbors(adata_invalid, radius=[0, 1.5])

    adata_invalid.obsm["spatial"] = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 4]])

    with pytest.raises(AssertionError):
        novae.utils.spatial_neighbors(adata_invalid, radius=[0, 1.5])

    with pytest.raises(AssertionError):
        novae.utils.spatial_neighbors(adata_invalid, radius=2, technology="unknown")

    with pytest.raises(AssertionError):
        novae.utils.spatial_neighbors(adata_invalid, radius=1, technology="cosmx")


def test_to_adjacency_local():
    adjancency_local = _to_adjacency_local(adata.obsp["spatial_connectivities"], 1)

    assert (
        (adjancency_local.todense() > 0)
        == np.array([
            [True, True, False, False, True],
            [True, True, True, False, True],
            [False, True, True, False, True],
            [False, False, False, True, False],
            [True, True, True, False, True],
        ])
    ).all()

    adjancency_local = _to_adjacency_local(adata.obsp["spatial_connectivities"], 2)

    assert (
        (adjancency_local.todense() > 0)
        == np.array([
            [True, True, True, False, True],
            [True, True, True, False, True],
            [True, True, True, False, True],
            [False, False, False, False, False],  # unconnected node => no local connections with n_hop >= 2
            [True, True, True, False, True],
        ])
    ).all()

    adjancency_local = _to_adjacency_local(adata_line.obsp["spatial_connectivities"], 1)

    assert (
        (adjancency_local.todense() > 0)
        == np.array([
            [True, True, False, False, False, False],
            [True, True, True, False, False, False],
            [False, True, True, False, False, False],
            [False, False, False, True, True, False],
            [False, False, False, True, True, False],
            [False, False, False, False, False, True],
        ])
    ).all()

    adjancency_local = _to_adjacency_local(adata_line.obsp["spatial_connectivities"], 2)

    assert (
        (adjancency_local.todense() > 0)
        == np.array([
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [False, False, False, True, True, False],
            [False, False, False, True, True, False],
            [False, False, False, False, False, False],  # unconnected node => no local connections with n_hop >= 2
        ])
    ).all()


def test_to_adjacency_view():
    adjancency_view = _to_adjacency_view(adata.obsp["spatial_connectivities"], 2)

    assert (
        (adjancency_view.todense() > 0)
        == np.array([
            [False, False, True, False, False],
            [False, False, False, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ])
    ).all()

    adjancency_view = _to_adjacency_view(adata.obsp["spatial_connectivities"], 3)

    assert adjancency_view.sum() == 0

    adjancency_view = _to_adjacency_view(adata_line.obsp["spatial_connectivities"], 1)

    assert (
        (adjancency_view.todense() > 0)
        == np.array([
            [False, True, False, False, False, False],
            [True, False, True, False, False, False],
            [False, True, False, False, False, False],
            [False, False, False, False, True, False],
            [False, False, False, True, False, False],
            [False, False, False, False, False, False],
        ])
    ).all()

    adjancency_view = _to_adjacency_view(adata_line.obsp["spatial_connectivities"], 2)

    assert (
        (adjancency_view.todense() > 0)
        == np.array([
            [False, False, True, False, False, False],
            [False, False, False, False, False, False],
            [True, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
        ])
    ).all()

    adjancency_view = _to_adjacency_view(adata_line.obsp["spatial_connectivities"], 3)

    assert adjancency_view.sum() == 0


def test_check_slide_name_key():
    obs = pd.DataFrame({Keys.SLIDE_ID: ["slide1", "slide2"], "name": ["sample1", "sample2"]})
    adata = AnnData(obs=obs)

    assert novae.utils.check_slide_name_key(adata, None) == Keys.SLIDE_ID
    novae.utils.check_slide_name_key(adata, "name")

    obs = pd.DataFrame({Keys.SLIDE_ID: ["slide1", "slide2", "slide2"], "name": ["sample1", "sample2", "sample3"]})
    adata2 = AnnData(obs=obs)

    assert novae.utils.check_slide_name_key(adata2, None) == Keys.SLIDE_ID
    with pytest.raises(AssertionError):
        novae.utils.check_slide_name_key(adata2, "name")

    del adata2.obs[Keys.SLIDE_ID]
    with pytest.raises(AssertionError):
        novae.utils.check_slide_name_key(adata2, None)


def test_change_n_hops():
    adata = novae.data.toy_dataset()[0]
    novae.spatial_neighbors(adata)

    model = novae.Novae(adata)
    model._init_datamodule()

    assert adata.uns[Keys.NOVAE_UNS]["n_hops_local"] == adata.uns[Keys.NOVAE_UNS]["n_hops_view"] == 2

    mean_adj_view = adata.obsp[Keys.ADJ_VIEW].getnnz(axis=0).mean()
    mean_adj_local = adata.obsp[Keys.ADJ_LOCAL].getnnz(axis=1).mean()

    model = novae.Novae(adata, n_hops_local=3, n_hops_view=4)
    model._init_datamodule()

    assert adata.uns[Keys.NOVAE_UNS]["n_hops_local"] == 3
    assert adata.uns[Keys.NOVAE_UNS]["n_hops_view"] == 4

    mean_adj_view2 = adata.obsp[Keys.ADJ_VIEW].getnnz(axis=0).mean()
    mean_adj_local2 = adata.obsp[Keys.ADJ_LOCAL].getnnz(axis=1).mean()

    assert mean_adj_view2 / mean_adj_view > 1.5
    assert mean_adj_local2 / mean_adj_local > 1.5


def test_new_distance_calculation() -> None:
    coords = np.random.rand(40, 2)

    N = coords.shape[0]

    tri = Delaunay(coords)
    indptr, indices = tri.vertex_neighbor_vertices

    dists = np.array(
        list(
            chain(
                *(
                    euclidean_distances(coords[indices[indptr[i] : indptr[i + 1]], :], coords[np.newaxis, i, :])
                    for i in range(N)
                    if len(indices[indptr[i] : indptr[i + 1]])
                )
            )
        )
    ).squeeze()
    Dst = csr_matrix((dists, indices, indptr), shape=(N, N))

    rows = np.repeat(np.arange(N), np.diff(indptr))
    dists = np.linalg.norm(coords[rows] - coords[indices], axis=1)
    Dst2 = csr_matrix((dists, indices, indptr), shape=(N, N))

    assert (Dst.indices == Dst2.indices).all()
    assert (Dst.indptr == Dst2.indptr).all()

    assert np.allclose(Dst.data, Dst2.data)
