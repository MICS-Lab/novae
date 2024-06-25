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


def test_invalid_build():
    adata_invalid = anndata.AnnData(obs=pd.DataFrame(index=["0", "1", "2"]))

    with pytest.raises(AssertionError):
        novae.utils.spatial_neighbors(adata_invalid, radius=[0, 1.5])

    adata_invalid.obsm["spatial"] = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 4]])

    with pytest.raises(AssertionError):
        novae.utils.spatial_neighbors(adata_invalid, radius=[0, 1.5])
