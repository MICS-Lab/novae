from __future__ import annotations

import numpy as np
from anndata import AnnData

from .._constants import ADJ, EPS


def mean_pide_score(adata: AnnData | list[AnnData], obs_key: str, slide_key: str = None) -> float:
    """Mean PIDE over all slides. A low score indicates a great domain continuity."""
    return np.mean(
        [pide_score(adata_slide, obs_key) for adata_slide in _iter_uid(adata, slide_key)]
    )


def pide_score(adata: AnnData, obs_key: str) -> float:
    """Percentage of inter-domain edges (PIDE). A low score indicates a great domain continuity."""
    i_left, i_right = adata.obsp[ADJ].nonzero()
    classes_left, classes_right = adata.obs.iloc[i_left][obs_key], adata.obs.iloc[i_right][obs_key]
    return (classes_left.values != classes_right.values).mean()


def jensen_shannon_divergence(
    adata: AnnData | list[AnnData], obs_key: str, slide_key: str = None
) -> float:
    """Jensen-Shannon divergence (JSD) over all slides

    Args:
        adata: One or a list of AnnData object(s)
        obs_key: The key containing the clusters
        slide_key: The slide ID obs key

    Returns:
        A float corresponding to the JSD
    """
    distributions = [
        adata_slide.obs[obs_key].value_counts(sort=False).values
        for adata_slide in _iter_uid(adata, slide_key, obs_key)
    ]

    return _jensen_shannon_divergence(np.array(distributions))


def _jensen_shannon_divergence(distributions: np.ndarray) -> float:
    """Compute the Jensen-Shannon divergence (JSD) for a multiple probability distributions.

    The lower the score, the better distribution of clusters among the different batches.

    Args:
        distributions: An array of shape (B x C), where B is the number of batches, and C is the number of clusters. For each batch, it contains the percentage of each cluster among cells.

    Returns:
        A float corresponding to the JSD
    """
    distributions = distributions / distributions.sum(1)[:, None]
    mean_distribution = np.mean(distributions, 0)

    return _entropy(mean_distribution) - np.mean([_entropy(dist) for dist in distributions])


def _entropy(distribution: np.ndarray) -> float:
    """Shannon entropy

    Args:
        distribution: An array of probabilities (should sum to one)

    Returns:
        The Shannon entropy
    """
    return -(distribution * np.log(distribution + EPS)).sum()


def _iter_uid(adatas: AnnData | list[AnnData], slide_key: str | None, obs_key: str | None = None):
    if isinstance(adatas, AnnData):
        adatas = [adatas]

    if obs_key is not None:
        categories = set.union(*[set(adata.obs[obs_key].unique().dropna()) for adata in adatas])
        for adata in adatas:
            adata.obs[obs_key] = (
                adata.obs[obs_key].astype("category").cat.set_categories(categories)
            )

    for adata in adatas:
        if slide_key is not None:
            for slide_id in adata.obs[slide_key].unique():
                yield adata[adata.obs[slide_key] == slide_id]
        else:
            yield adata