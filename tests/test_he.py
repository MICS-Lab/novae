import numpy as np
import pytest
import sopa

import novae


def test_he_embeddings():
    sdata1 = sopa.io.toy_dataset(as_output=True, genes=50)
    sdata2 = sopa.io.toy_dataset(as_output=True, genes=50)

    novae.data.compute_histo_embeddings(sdata1, "dummy", patch_overlap_ratio=0.6)
    novae.data.compute_histo_pca(sdata1, n_components=2)

    X: np.ndarray = sdata1["table"].obsm["histo_embeddings"]
    assert (X.mean(0) ** 2).max() < 0.05 and ((1 - X.std(0)) ** 2).max() < 0.05

    adatas = [sdata1["table"], sdata2["table"]]
    novae.spatial_neighbors(adatas)

    with pytest.raises(AssertionError):
        novae.Novae(adatas)  # only one adata with H&E embeddings

    model = novae.Novae([adatas[0], adatas[0]])

    assert model.mode.multimodal  # True because both adatas have H&E embeddings

    novae.settings.disable_multimodal = True

    model = novae.Novae(adatas)

    assert not model.mode.multimodal

    model = novae.Novae([adatas[0], adatas[0]])

    assert not model.mode.multimodal  # False because disabled
