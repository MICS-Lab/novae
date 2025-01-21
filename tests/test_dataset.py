import numpy as np
import pytest

import novae
from novae import utils
from novae._constants import Keys, Nums

N_PANELS = 2
N_SLIDES_PER_PANEL = 3


adatas = novae.utils.toy_dataset(n_panels=N_PANELS, n_slides_per_panel=N_SLIDES_PER_PANEL)

single_adata = adatas[0]


def test_raise_invalid_slide_id():
    with pytest.raises(AssertionError):
        novae.utils.spatial_neighbors(adatas, slide_key="key_not_in_obs")


def test_single_panel():
    novae.utils.spatial_neighbors(single_adata)
    model = novae.Novae(single_adata)
    model._datamodule = model._init_datamodule()

    assert len(model.dataset.slides_metadata) == 1
    assert model.dataset.obs_ilocs is not None


def test_single_panel_slide_key():
    Nums.BATCH_INTER_SLIDE_RATIO = 1
    novae.utils.spatial_neighbors(single_adata, slide_key="slide_key")
    model = novae.Novae(single_adata)
    model._datamodule = model._init_datamodule()

    assert len(model.dataset.slides_metadata) == N_SLIDES_PER_PANEL
    assert model.dataset.obs_ilocs is not None

    _ensure_batch_same_slide(model)
    Nums.BATCH_INTER_SLIDE_RATIO = 0.5


def test_multi_panel():
    Nums.BATCH_INTER_SLIDE_RATIO = 1
    novae.utils.spatial_neighbors(adatas)
    model = novae.Novae(adatas)
    model._datamodule = model._init_datamodule()

    assert len(model.dataset.slides_metadata) == N_PANELS
    assert model.dataset.obs_ilocs is None

    _ensure_batch_same_slide(model)
    Nums.BATCH_INTER_SLIDE_RATIO = 0.5


def test_multi_panel_slide_key():
    Nums.BATCH_INTER_SLIDE_RATIO = 1
    novae.utils.spatial_neighbors(adatas, slide_key="slide_key")
    model = novae.Novae(adatas)
    model._datamodule = model._init_datamodule()

    assert len(model.dataset.slides_metadata) == N_PANELS * N_SLIDES_PER_PANEL
    assert model.dataset.obs_ilocs is None

    _ensure_batch_same_slide(model)
    Nums.BATCH_INTER_SLIDE_RATIO = 0.5


def _ensure_batch_same_slide(model: novae.Novae):
    n_obs_dataset = model.dataset.shuffled_obs_ilocs.shape[0]
    assert n_obs_dataset % model.hparams.batch_size == 0

    for batch_index in range(n_obs_dataset // model.hparams.batch_size):
        sub_obs_ilocs = model.dataset.shuffled_obs_ilocs[
            batch_index * model.hparams.batch_size : (batch_index + 1) * model.hparams.batch_size
        ]
        unique_adata_indices = np.unique(sub_obs_ilocs[:, 0])
        assert len(unique_adata_indices) == 1
        slide_ids = adatas[unique_adata_indices[0]].obs[Keys.SLIDE_ID].iloc[sub_obs_ilocs[:, 1]]
        assert len(np.unique(slide_ids)) == 1


def test_cross_slide_batches():
    adatas = novae.utils.toy_dataset(n_panels=2, n_slides_per_panel=2, xmax=400)

    novae.utils.spatial_neighbors(adatas)
    model = novae.Novae(adatas)

    model._datamodule = model._init_datamodule()
    loader = model.datamodule.train_dataloader()

    n_batches, n_inter_slide_batches = 0, 0
    has_started_inter_slide = False

    for batch in loader:
        n_batches += 1
        slide_id = utils.get_mini_batch_slide_id(batch)
        if slide_id is not None:
            assert (batch["main"].genes_indices == batch["main"].genes_indices.to(float).mean(0)).all()
            n_inter_slide_batches += 1
            has_started_inter_slide = True
        else:
            assert not has_started_inter_slide

    assert n_inter_slide_batches > 0 and n_batches > n_inter_slide_batches
