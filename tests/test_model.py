import json

import anndata
import numpy as np
import pandas as pd
import pytest
import torch

import novae
from novae._constants import Keys
from novae.utils._data import TRUE_GENE_NAMES

adatas = novae.utils.dummy_dataset(
    n_panels=2,
    n_slides_per_panel=2,
    xmax=100,
    n_domains=2,
    compute_spatial_neighbors=True,
)
adata = adatas[0]


def _generate_fake_scgpt_inputs():
    gene_names = TRUE_GENE_NAMES[:100]
    indices = [7, 4, 3, 0, 1, 2, 9, 5, 6, 8] + list(range(10, 100))

    vocab = dict(zip(gene_names, indices))

    with open("tests/vocab.json", "w") as f:
        json.dump(vocab, f, indent=4)

    torch.save({"encoder.embedding.weight": torch.randn(len(vocab), 16)}, "tests/best_model.pt")


_generate_fake_scgpt_inputs()


def test_load_wandb_artifact():
    novae.Novae._load_wandb_artifact("novae/novae/model-4i8e9g2v:v17")


def test_load_huggingface_model():
    novae.Novae.from_pretrained("MICS-Lab/novae-test")


def test_train():
    adatas = novae.utils.dummy_dataset(
        n_panels=2,
        n_slides_per_panel=2,
        xmax=100,
        n_domains=2,
    )

    model = novae.Novae(adatas)

    with pytest.raises(AssertionError):  # should raise an error because the model has not been trained
        model.compute_representations()

    model.fit(max_epochs=1)
    model.compute_representations()
    obs_key = model.assign_domains(level=2)

    adatas[0].obs.iloc[0][obs_key] = np.nan

    svg1 = novae.monitor.mean_svg_score(adatas, obs_key=obs_key)
    fide1 = novae.monitor.mean_fide_score(adatas, obs_key=obs_key)
    jsd1 = novae.monitor.jensen_shannon_divergence(adatas, obs_key=obs_key)

    svg2 = novae.monitor.mean_svg_score(adatas, obs_key=obs_key, slide_key="slide_key")
    fide2 = novae.monitor.mean_fide_score(adatas, obs_key=obs_key, slide_key="slide_key")
    jsd2 = novae.monitor.jensen_shannon_divergence(adatas, obs_key=obs_key, slide_key="slide_key")

    assert svg1 == -1000 or svg1 != svg2
    assert fide1 != fide2
    assert jsd1 != jsd2

    adatas[0].write_h5ad("tests/test.h5ad")  # ensures the output can be saved


@pytest.mark.parametrize("slide_key", [None, "slide_key"])
def test_representation_single_panel(slide_key: str | None):
    adata = novae.utils.dummy_dataset(
        n_panels=1,
        n_slides_per_panel=2,
        xmax=100,
        n_domains=2,
    )[0]

    model = novae.Novae(adata, slide_key=slide_key)
    model._datamodule = model._init_datamodule()
    model.mode.trained = True

    model.compute_representations()

    domains = adata.obs[Keys.LEAVES].copy()

    model.compute_representations()

    assert domains.equals(adata.obs[Keys.LEAVES])

    model.compute_representations([adata], slide_key=slide_key)

    assert domains.equals(adata.obs[Keys.LEAVES])

    if slide_key is not None:
        sids = adata.obs[slide_key].unique()
        adatas = [adata[adata.obs[slide_key] == sid] for sid in sids]

        model.compute_representations(adatas)

        adata_concat = anndata.concat(adatas)

        assert domains.equals(adata_concat.obs[Keys.LEAVES].loc[domains.index])

        model.compute_representations(adatas, slide_key=slide_key)

        adata_concat = anndata.concat(adatas)

        assert domains.equals(adata_concat.obs[Keys.LEAVES].loc[domains.index])


@pytest.mark.parametrize("slide_key", [None, "slide_key"])
def test_representation_multi_panel(slide_key: str | None):
    adatas = novae.utils.dummy_dataset(
        n_panels=3,
        n_slides_per_panel=2,
        xmax=100,
        n_domains=2,
    )

    model = novae.Novae(adatas, slide_key=slide_key)
    model._datamodule = model._init_datamodule()
    model.mode.trained = True

    model.compute_representations()

    domains_series = pd.concat([adata.obs[Keys.LEAVES].copy() for adata in adatas])

    model.compute_representations(adatas, slide_key=slide_key)

    domains_series2 = pd.concat([adata.obs[Keys.LEAVES].copy() for adata in adatas])

    assert domains_series.equals(domains_series2.loc[domains_series.index])

    adata_split = [
        adata[adata.obs[Keys.SLIDE_ID] == sid].copy() for adata in adatas for sid in adata.obs[Keys.SLIDE_ID].unique()
    ]

    model.compute_representations(adata_split)

    domains_series2 = pd.concat([adata.obs[Keys.LEAVES] for adata in adata_split])

    assert domains_series.equals(domains_series2.loc[domains_series.index])


@pytest.mark.parametrize("slide_key", [None, "slide_key"])
@pytest.mark.parametrize("scgpt_model_dir", [None, "tests"])
def test_saved_model_identical(slide_key: str | None, scgpt_model_dir: str | None):
    adata = novae.utils.dummy_dataset(
        n_panels=1,
        n_slides_per_panel=2,
        xmax=100,
        n_domains=2,
    )[0]

    # using weird parameters
    model = novae.Novae(
        adata,
        slide_key=slide_key,
        embedding_size=67,
        output_size=78,
        n_hops_local=4,
        n_hops_view=3,
        heads=2,
        hidden_size=46,
        num_layers=3,
        batch_size=345,
        temperature=0.13,
        num_prototypes=212,
        panel_subset_size=0.62,
        background_noise_lambda=7.7,
        sensitivity_noise_std=0.042,
        scgpt_model_dir=scgpt_model_dir,
    )

    model._datamodule = model._init_datamodule()
    model.mode.trained = True

    model.compute_representations()
    model.assign_domains()

    domains = adata.obs[Keys.LEAVES].copy()
    representations = adata.obsm[Keys.REPR].copy()

    model.save_pretrained("tests/test_model")

    new_model = novae.Novae.from_pretrained("tests/test_model")

    new_model.compute_representations(adata, slide_key=slide_key)
    new_model.assign_domains(adata)

    assert (adata.obsm[Keys.REPR] == representations).all()
    assert domains.equals(adata.obs[Keys.LEAVES])
