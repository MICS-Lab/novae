from unittest.mock import MagicMock

import anndata
import numpy as np
import pytest

import novae
from novae._constants import Keys
from novae.utils._validate import log

N_PANELS = 2
N_SLIDES_PER_PANEL = 3


adatas = novae.data.toy_dataset(n_panels=N_PANELS, n_slides_per_panel=N_SLIDES_PER_PANEL)

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
    novae.utils.spatial_neighbors(single_adata, slide_key="slide_key")
    model = novae.Novae(single_adata)
    model._datamodule = model._init_datamodule()

    assert len(model.dataset.slides_metadata) == N_SLIDES_PER_PANEL
    assert model.dataset.obs_ilocs is not None

    _ensure_batch_same_slide(model)


def test_multi_panel():
    novae.utils.spatial_neighbors(adatas)
    model = novae.Novae(adatas)
    model._datamodule = model._init_datamodule()

    assert len(model.dataset.slides_metadata) == N_PANELS
    assert model.dataset.obs_ilocs is None

    _ensure_batch_same_slide(model)


def test_multi_panel_slide_key():
    novae.utils.spatial_neighbors(adatas, slide_key="slide_key")
    model = novae.Novae(adatas)
    model._datamodule = model._init_datamodule()

    assert len(model.dataset.slides_metadata) == N_PANELS * N_SLIDES_PER_PANEL
    assert model.dataset.obs_ilocs is None

    _ensure_batch_same_slide(model)


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


def test_multi_slide_one_adata():
    adatas = novae.data.toy_dataset(n_drop=0)

    for i, adata in enumerate(adatas):
        adata.obs["slide_id"] = f"slide_{i}"

    adata = anndata.concat(adatas)

    novae.spatial_neighbors(adata, slide_key="slide_id")

    model = novae.Novae(adata)
    datamodule = model._init_datamodule()

    unique_ids = set()
    for batch in datamodule.train_dataloader():
        slide_ids = set(batch["main"]["slide_id"])
        assert len(slide_ids) == 1
        unique_ids.update(slide_ids)

    assert len(unique_ids) == len(adatas)


def test_uses_scale_to_microns():
    adatas = novae.toy_dataset(n_panels=1, compute_spatial_neighbors=True)

    for adata in adatas:
        adata.obsp[Keys.ADJ].data *= 10

    log.warning = MagicMock()

    model = novae.Novae(adatas, embedding_size=40)
    datamodule = model._init_datamodule()
    assert next(iter(datamodule.train_dataloader()))["main"].edge_attr.mean() > 10

    log.warning.assert_called_once()

    log.warning = MagicMock()

    novae.settings.scale_to_microns = 0.1
    model = novae.Novae(adatas, embedding_size=40)
    datamodule = model._init_datamodule()
    assert next(iter(datamodule.train_dataloader()))["main"].edge_attr.mean() < 2

    novae.settings.scale_to_microns = 1

    log.warning.assert_not_called()
