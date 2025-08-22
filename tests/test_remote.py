import novae


def test_load_dataset_filters():
    total_samples = len(novae.load_dataset(dry_run=True))
    assert total_samples >= 95

    n_breast_samples = len(novae.load_dataset(dry_run=True, tissue="breast"))
    assert 7 <= n_breast_samples < total_samples

    n_human_samples = len(novae.load_dataset(dry_run=True, species="human"))
    assert 75 <= n_human_samples < total_samples

    n_cosmx_samples = len(novae.load_dataset(dry_run=True, technology="cosmx"))
    assert 11 <= n_cosmx_samples < total_samples

    assert len(novae.load_dataset(dry_run=True, species="human", top_k=2)) == 2

    mgc_dataset_size = len(
        novae.load_dataset(
            tissue="head_and_neck",
            custom_filter=lambda df: df["n_proteins"] > 0,
            technology="cosmx",
            species="human",
            dry_run=True,
        )
    )

    assert mgc_dataset_size == 8

    assert len(novae.load_dataset(pattern="HumanOvarianCancerPatient*", dry_run=True)) == 4


def test_load_small_dataset():
    res = novae.load_dataset(pattern="Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs")

    assert len(res) == 1

    adata = res[0]

    assert adata.shape == (24406, 313)
    assert 5 <= adata.X.max() <= 6
    assert "spatial" in adata.obsm
    assert adata.layers["counts"].max() == 94
