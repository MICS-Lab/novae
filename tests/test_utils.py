import anndata
import numpy as np
import pandas as pd
import pytest

import novae

from .test_metrics import adatas

adata = adatas[0]

true_connectivities = np.array(
    [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
    ]
)


def test_build():
    connectivities = adata.obsp["spatial_connectivities"]

    assert connectivities.shape[0] == adata.n_obs

    assert (connectivities.A == true_connectivities).all()


def test_build_pixel_size():
    adata_pixel = adata.copy()
    novae.utils.spatial_neighbors(adata_pixel, radius=5, pixel_size=10)

    connectivities = adata_pixel.obsp["spatial_connectivities"]
    assert (connectivities.A == 0).all()

    adata_pixel = adata.copy()
    novae.utils.spatial_neighbors(adata_pixel, radius=15, pixel_size=10)

    connectivities = adata_pixel.obsp["spatial_connectivities"]
    assert (connectivities.A == true_connectivities).all()

    # this should raise an error, because the function is being called twice with pixel_size
    with pytest.raises(AssertionError):
        novae.utils.spatial_neighbors(adata_pixel, radius=15, pixel_size=10)


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
        novae.utils.spatial_neighbors(adata_invalid, radius=1, technology="cosmx", pixel_size=0.1)
