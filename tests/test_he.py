import numpy as np
import pytest
import sopa
import torch

import novae


def test_he_embeddings():
    sdata1 = sopa.io.toy_dataset(as_output=True, genes=50)
    sdata2 = sopa.io.toy_dataset(as_output=True, genes=50)

    novae.data.compute_histo_embeddings(sdata1, "dummy", patch_overlap_ratio=0.6)
    novae.data.compute_histo_pca(sdata1, n_components=2)

    X: np.ndarray = sdata1["table"].obsm["histo_embeddings"]
    assert (X.mean(0) ** 2).max() < 0.05 and ((1 - X.std(0)) ** 2).max() < 0.05

    adatas = [sdata1["table"], sdata2["table"]]
    adata = adatas[0]
    novae.spatial_neighbors(adatas)

    with pytest.raises(AssertionError):
        novae.Novae(adatas, histo_embedding_size=2)  # only one adata with H&E embeddings

    model = novae.Novae(adata, histo_embedding_size=2)

    assert model.mode.multimodal  # True because adata has H&E embeddings

    w = model.encoder.mlp_fusion[0].weight.data.clone()

    model.fit(max_epochs=1)
    model.compute_representations()

    model.save_pretrained("tests/test_he_model")
    assert novae.Novae.from_pretrained("tests/test_he_model").mode.multimodal  # recover the multimodal mode

    w2 = model.encoder.mlp_fusion[0].weight.data.clone()

    assert not torch.allclose(w, w2)

    with pytest.raises(ValueError):
        model.compute_representations(adatas[1])

    novae.settings.disable_multimodal = True

    w = model.encoder.mlp_fusion[0].weight.data.clone()
    model.fit(max_epochs=1)
    w2 = model.encoder.mlp_fusion[0].weight.data
    assert (w == w2).all()  # no change in weights

    model = novae.Novae(adatas)

    assert not model.mode.multimodal

    w = model.encoder.mlp_fusion[0].weight.data.clone()
    model.fit(max_epochs=1)
    w2 = model.encoder.mlp_fusion[0].weight.data
    assert (w == w2).all()  # no change in weights

    model = novae.Novae([adatas[0], adatas[0]])

    assert not model.mode.multimodal  # False because disabled
