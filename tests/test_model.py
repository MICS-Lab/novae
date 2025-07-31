import json

import anndata
import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

import novae
from novae._constants import Keys
from novae.data._load._toy import GENE_NAMES_SUBSET

adatas = novae.data.toy_dataset(
    n_panels=2,
    n_slides_per_panel=2,
    xmax=100,
    n_domains=2,
    compute_spatial_neighbors=True,
)
adata = adatas[0]


def _generate_fake_scgpt_inputs():
    gene_names = GENE_NAMES_SUBSET[:100]
    indices = [7, 4, 3, 0, 1, 2, 9, 5, 6, 8, *list(range(10, 100))]

    vocab = dict(zip(gene_names, indices))

    with open("tests/vocab.json", "w") as f:
        json.dump(vocab, f, indent=4)

    torch.save({"encoder.embedding.weight": torch.randn(len(vocab), 16)}, "tests/best_model.pt")


_generate_fake_scgpt_inputs()


# def test_load_wandb_artifact():
#     novae.Novae._load_wandb_artifact("novae/novae/model-4i8e9g2v:v17")


def test_load_huggingface_model():
    model = novae.Novae.from_pretrained("MICS-Lab/novae-test")

    assert model.cell_embedder.embedding.weight.requires_grad is False


def test_train():
    adatas = novae.data.toy_dataset(
        n_panels=2,
        n_slides_per_panel=2,
        xmax=60,
        n_domains=2,
    )

    novae.utils.spatial_neighbors(adatas)
    model = novae.Novae(adatas, num_prototypes=10)

    with pytest.raises(AssertionError):  # should raise an error because the model has not been trained
        model.compute_representations()

    model.fit(max_epochs=3)
    model.compute_representations()
    model.compute_representations(num_workers=2)

    # obs_key = model.assign_domains(n_domains=2)
    obs_key = model.assign_domains(level=2)

    model.batch_effect_correction()

    adatas[0].obs.iloc[0][obs_key] = np.nan

    novae.monitor.mean_fide_score(adatas, obs_key=obs_key)
    novae.monitor.jensen_shannon_divergence(adatas, obs_key=obs_key)

    novae.monitor.mean_fide_score(adatas, obs_key=obs_key, slide_key="slide_key")
    novae.monitor.jensen_shannon_divergence(adatas, obs_key=obs_key, slide_key="slide_key")

    adatas[0].write_h5ad("tests/test.h5ad")  # ensures the output can be saved

    model.compute_representations(adatas, zero_shot=True)

    with pytest.raises(AssertionError):
        model.fine_tune(adatas, max_epochs=1)

    model.mode.pretrained = True

    model.fine_tune(adatas, max_epochs=1)


@pytest.mark.parametrize("slide_key", [None, "slide_key"])
def test_representation_single_panel(slide_key: str | None):
    adata = novae.data.toy_dataset(
        n_panels=1,
        n_slides_per_panel=2,
        xmax=100,
        n_domains=2,
    )[0]

    novae.utils.spatial_neighbors(adata, slide_key=slide_key)

    model = novae.Novae(adata)
    model._datamodule = model._init_datamodule()
    model.mode.trained = True

    model.compute_representations()

    domains = adata.obs[Keys.LEAVES].copy()

    model.compute_representations()

    assert domains.equals(adata.obs[Keys.LEAVES])

    novae.utils.spatial_neighbors(adata, slide_key=slide_key)
    model.compute_representations([adata])

    assert domains.equals(adata.obs[Keys.LEAVES])

    if slide_key is not None:
        sids = adata.obs[slide_key].unique()
        adatas = [adata[adata.obs[slide_key] == sid] for sid in sids]

        novae.utils.spatial_neighbors(adatas)
        model.compute_representations(adatas)

        adata_concat = anndata.concat(adatas)

        assert domains.equals(adata_concat.obs[Keys.LEAVES].loc[domains.index])

        novae.utils.spatial_neighbors(adatas, slide_key=slide_key)
        model.compute_representations(adatas)

        adata_concat = anndata.concat(adatas)

        assert domains.equals(adata_concat.obs[Keys.LEAVES].loc[domains.index])


@pytest.mark.parametrize("slide_key", [None, "slide_key"])
def test_representation_multi_panel(slide_key: str | None):
    adatas = novae.data.toy_dataset(
        n_panels=3,
        n_slides_per_panel=2,
        xmax=100,
        n_domains=2,
    )

    novae.utils.spatial_neighbors(adatas, slide_key=slide_key)
    model = novae.Novae(adatas)
    model._datamodule = model._init_datamodule()
    model.mode.trained = True

    model.compute_representations()

    domains_series = pd.concat([adata.obs[Keys.LEAVES].copy() for adata in adatas])

    novae.utils.spatial_neighbors(adatas, slide_key=slide_key)
    model.compute_representations(adatas)

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
    adata = novae.data.toy_dataset(
        n_panels=1,
        n_slides_per_panel=2,
        xmax=100,
        n_domains=2,
    )[0]

    # using weird parameters
    novae.utils.spatial_neighbors(adata, slide_key=slide_key)
    model = novae.Novae(
        adata,
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

    assert model.cell_embedder.embedding.weight.requires_grad is (scgpt_model_dir is None)

    model._datamodule = model._init_datamodule()
    model.mode.trained = True

    model.compute_representations()
    model.assign_domains()

    domains = adata.obs[Keys.LEAVES].copy()
    representations = adata.obsm[Keys.REPR].copy()

    model.save_pretrained("tests/test_model")

    new_model = novae.Novae.from_pretrained("tests/test_model")

    novae.utils.spatial_neighbors(adata, slide_key=slide_key)
    new_model.compute_representations(adata)
    new_model.assign_domains(adata)

    assert (adata.obsm[Keys.REPR] == representations).all()
    assert domains.equals(adata.obs[Keys.LEAVES])

    for name, param in model.named_parameters():
        assert torch.equal(param, new_model.state_dict()[name])


def test_safetensors_parameters_names():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    local_file = hf_hub_download(repo_id="MICS-Lab/novae-human-0", filename="model.safetensors")
    with safe_open(local_file, framework="pt", device="cpu") as f:
        pretrained_model_names = f.keys()

    model = novae.Novae(adata)

    actual_names = [name for name, _ in model.named_parameters()]

    # TODO: remove this after MICS-Lab/novae-human-1 release (and update where it is laoded)
    actual_names = [name for name in actual_names if not name.startswith("encoder.mlp_fusion")]

    assert set(pretrained_model_names) == set(actual_names)


def test_reset_clusters_zero_shot():
    adata = novae.data.toy_dataset()[0]

    novae.utils.spatial_neighbors(adata)

    model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")

    model.compute_representations(adata, zero_shot=True)
    clusters_levels = model.swav_head.clusters_levels.copy()

    adata = adata[:550].copy()

    model.compute_representations(adata, zero_shot=True)

    assert not (model.swav_head.clusters_levels == clusters_levels).all()


def test_init_prototypes():
    model = novae.Novae(adata, num_prototypes=20)

    prototypes = model.swav_head.prototypes.data.clone()
    model.init_prototypes(adata)
    assert (model.swav_head.prototypes.data != prototypes).all()


def test_var_name_subset():
    adata = AnnData(np.random.rand(10, 30))
    adata.var_names = [f"GENE{i}" for i in range(30)]
    adata.obsm["spatial"] = np.random.randn(10, 2)

    novae.spatial_neighbors(adata)

    selected_genes = ["gene1"] + [f"GENE{i}" for i in range(5, 25)]  # check case insensitivity
    model = novae.Novae(adata, var_names=selected_genes, embedding_size=10)

    assert model.hparams.var_names == [x.lower() for x in selected_genes]
