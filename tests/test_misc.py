from __future__ import annotations

import novae
from novae._constants import Nums

adata = novae.utils.dummy_dataset(xmax=200)[0]


def test_settings():
    novae.settings.disable_lazy_loading()

    model = novae.Novae(adata)
    model._datamodule = model._init_datamodule()
    assert model.dataset.anndata_torch.tensors is not None

    novae.settings.enable_lazy_loading(n_obs_threshold=100)
    model = novae.Novae(adata)
    model._datamodule = model._init_datamodule()
    assert model.dataset.anndata_torch.tensors is None

    novae.settings.warmup_epochs = 2
    assert novae.settings.warmup_epochs == Nums.WARMUP_EPOCHS


def test_repr():
    model = novae.Novae(adata)

    repr(model)
    repr(model.mode)