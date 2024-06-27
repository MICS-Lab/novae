import anndata
import numpy as np
import pandas as pd
from anndata import AnnData
from pytest import approx

import novae
from novae.monitor.eval import _jensen_shannon_divergence, entropy

domains = [
    ["D1", "D2", "D3", "D4", "D5"],
    ["D1", "D1", "D2", "D2", "D3"],
    ["D1", "D1", "D1", "D2", np.nan],
]

true_distributions = np.array(
    [
        [0.2] * 5,
        [0.4, 0.4, 0.2, 0, 0],
        [0.75, 0.25, 0, 0, 0],
    ]
)

true_jsd = _jensen_shannon_divergence(true_distributions)

spatial_coords = np.array(
    [
        [[0, 0], [0, 1], [0, 2], [3, 3], [1, 1]],
        [[0, 0], [0, 1], [0, 2], [3, 3], [1, 1]],
        [[0, 0], [0, 1], [0, 2], [3, 3], [1, 1]],
    ]
)


def _get_adata(i: int):
    values = domains[i]
    adata = AnnData(obs=pd.DataFrame({"domain": values}, index=[str(i) for i in range(len(values))]))
    adata.obs["slide_key"] = f"slide_{i}"
    adata.obsm["spatial"] = spatial_coords[i]
    novae.utils.spatial_neighbors(adata, radius=[0, 1.5])
    return adata


adatas = [_get_adata(i) for i in range(len(domains))]

adata_concat = anndata.concat(adatas)


def test_jensen_shannon_divergence():
    jsd = novae.monitor.jensen_shannon_divergence(adatas, "domain")

    assert jsd == approx(true_jsd)


def test_jensen_shannon_divergence_concat():
    jsd = novae.monitor.jensen_shannon_divergence(adata_concat, "domain", slide_key="slide_key")

    assert jsd == approx(true_jsd)


def test_jensen_shannon_divergence_manual():
    assert _jensen_shannon_divergence(np.ones((1, 5))) == approx(0.0)
    assert _jensen_shannon_divergence(np.ones((2, 5))) == approx(0.0)

    distribution = np.array(
        [
            [0.3, 0.2, 0.5],
            [0.1, 0.1, 0.8],
            [0.2, 0.3, 0.5],
            [0, 0, 1],
        ]
    )

    means = np.array([0.15, 0.15, 0.7])

    entropy_means = entropy(means)

    assert entropy_means == approx(1.18, rel=1e-2)

    jsd_manual = entropy_means - 0.25 * sum(entropy(d) for d in distribution)

    jsd = _jensen_shannon_divergence(distribution)

    assert jsd == approx(jsd_manual)


def test_fide_score():
    fide = novae.monitor.fide_score(adatas[0], "domain", n_classes=5)

    assert fide == 0

    fide = novae.monitor.fide_score(adatas[1], "domain", n_classes=3)

    assert fide == approx(0.4 / 3)

    fide = novae.monitor.fide_score(adatas[1], "domain", n_classes=5)

    assert fide == approx(0.4 / 5)

    fide = novae.monitor.fide_score(adatas[2], "domain", n_classes=1)

    assert fide == approx(1)

    fide = novae.monitor.fide_score(adatas[2], "domain", n_classes=5)

    assert fide == approx(1 / 5)
