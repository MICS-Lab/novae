import anndata
import numpy as np
import pandas as pd
import pytest

import novae
from novae._constants import Keys

adatas = novae.utils.dummy_dataset(
    n_panels=2,
    n_slides_per_panel=2,
    n_obs_per_domain=100,
    n_domains=2,
)
adata = adatas[0]


def test_load_old_model():
    # should tell the model from the checkpoint was trained with an old novae version
    with pytest.raises(ValueError):
        novae.Novae.load_from_wandb_artifact("novae/novae_swav/model-w9xjvjjy:v29")


def test_load_model():
    novae.Novae.load_from_wandb_artifact("novae/novae/model-4i8e9g2v:v17")


def test_train():
    adatas = novae.utils.dummy_dataset(
        n_panels=2,
        n_slides_per_panel=2,
        n_obs_per_domain=100,
        n_domains=2,
    )

    model = novae.Novae(adatas)
    model.fit(max_epochs=1)
    model.compute_representation()
    obs_key = model.assign_domains(k=2)

    adatas[0].obs.iloc[0][obs_key] = np.nan

    svg1 = novae.monitor.mean_svg_score(adatas, obs_key=obs_key)
    fide1 = novae.monitor.mean_fide_score(adatas, obs_key=obs_key)
    jsd1 = novae.monitor.jensen_shannon_divergence(adatas, obs_key=obs_key)

    svg2 = novae.monitor.mean_svg_score(adatas, obs_key=obs_key, slide_key="slide_key")
    fide2 = novae.monitor.mean_fide_score(adatas, obs_key=obs_key, slide_key="slide_key")
    jsd2 = novae.monitor.jensen_shannon_divergence(adatas, obs_key=obs_key, slide_key="slide_key")

    assert svg1 != svg2
    assert fide1 != fide2
    assert jsd1 != jsd2


@pytest.mark.parametrize("slide_key", [None, "slide_key"])
def test_representation_single_panel(slide_key: str | None):
    adata = novae.utils.dummy_dataset(
        n_panels=1,
        n_slides_per_panel=2,
        n_obs_per_domain=100,
        n_domains=2,
        compute_spatial_neighbors=False,
    )[0]

    model = novae.Novae(adata, slide_key=slide_key)
    model._datamodule = model._init_datamodule()

    model.compute_representation()

    niches = adata.obs[Keys.SWAV_CLASSES].copy()

    model.compute_representation()

    assert (niches == adata.obs[Keys.SWAV_CLASSES]).all()

    model.compute_representation([adata], slide_key=slide_key)

    assert (niches == adata.obs[Keys.SWAV_CLASSES]).all()

    if slide_key is not None:
        sids = adata.obs[slide_key].unique()
        adatas = [adata[adata.obs[slide_key] == sid] for sid in sids]

        model.compute_representation(adatas)

        adata_concat = anndata.concat(adatas)

        assert (niches == adata_concat.obs[Keys.SWAV_CLASSES].loc[niches.index]).all()

        model.compute_representation(adatas, slide_key=slide_key)

        adata_concat = anndata.concat(adatas)

        assert (niches == adata_concat.obs[Keys.SWAV_CLASSES].loc[niches.index]).all()


@pytest.mark.parametrize("slide_key", [None, "slide_key"])
def test_representation_multi_panel(slide_key: str | None):
    adatas = novae.utils.dummy_dataset(
        n_panels=3,
        n_slides_per_panel=2,
        n_obs_per_domain=100,
        n_domains=2,
        compute_spatial_neighbors=False,
    )

    model = novae.Novae(adatas, slide_key=slide_key)
    model._datamodule = model._init_datamodule()

    model.compute_representation()

    niches_series = pd.concat([adata.obs[Keys.SWAV_CLASSES].copy() for adata in adatas])

    model.compute_representation(adatas, slide_key=slide_key)

    niches_series2 = pd.concat([adata.obs[Keys.SWAV_CLASSES].copy() for adata in adatas])

    assert (niches_series2.loc[niches_series.index] == niches_series).all()

    adata_split = [
        adata[adata.obs[Keys.SLIDE_ID] == sid].copy() for adata in adatas for sid in adata.obs[Keys.SLIDE_ID].unique()
    ]

    model.compute_representation(adata_split)

    niches_series2 = pd.concat([adata.obs[Keys.SWAV_CLASSES] for adata in adata_split])

    assert (niches_series2.loc[niches_series.index] == niches_series).all()
