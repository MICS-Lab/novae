import logging

import numpy as np
from anndata import AnnData
from sklearn import metrics

from .._constants import Keys, Nums

log = logging.getLogger(__name__)


def mean_fide_score(
    adatas: AnnData | list[AnnData], obs_key: str, slide_key: str = None, n_classes: int | None = None
) -> float:
    """Mean FIDE score over all slides. A low score indicates a great domain continuity.

    Args:
        adatas: An `AnnData` object, or a list of `AnnData` objects.
        obs_key: Key of `adata.obs` containing the domains annotation.
        slide_key: Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.
        n_classes: Optional number of classes. This can be useful if not all classes are predicted, for a fair comparision.

    Returns:
        The FIDE score averaged for all slides.
    """
    return np.mean(
        [
            fide_score(adata, obs_key, n_classes=n_classes)
            for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key)
        ]
    )


def fide_score(adata: AnnData, obs_key: str, n_classes: int | None = None) -> float:
    """F1-score of intra-domain edges (FIDE). A high score indicates a great domain continuity.

    Note:
        The F1-score is computed for every class, then all F1-scores are averaged. If some classes
        are not predicted, the `n_classes` argument allows to pad with zeros before averaging the F1-scores.

    Args:
        adata: An `AnnData` object
        obs_key: Key of `adata.obs` containing the domains annotation.
        n_classes: Optional number of classes. This can be useful if not all classes are predicted, for a fair comparision.

    Returns:
        The FIDE score.
    """
    i_left, i_right = adata.obsp[Keys.ADJ].nonzero()
    classes_left, classes_right = adata.obs.iloc[i_left][obs_key].values, adata.obs.iloc[i_right][obs_key].values

    where_valid = ~classes_left.isna() & ~classes_right.isna()
    classes_left, classes_right = classes_left[where_valid], classes_right[where_valid]

    f1_scores = metrics.f1_score(classes_left, classes_right, average=None)

    if n_classes is None:
        return f1_scores.mean()

    assert n_classes >= len(f1_scores), f"Expected {n_classes:=}, but found {len(f1_scores)}, which is greater"

    return np.pad(f1_scores, (0, n_classes - len(f1_scores))).mean()


def jensen_shannon_divergence(adatas: AnnData | list[AnnData], obs_key: str, slide_key: str = None) -> float:
    """Jensen-Shannon divergence (JSD) over all slides

    Args:
        adatas: One or a list of AnnData object(s)
        obs_key: Key of `adata.obs` containing the domains annotation.
        slide_key: Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.

    Returns:
        The Jensen-Shannon divergence score for all slides
    """
    all_categories = set()
    for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key):
        all_categories.update(adata.obs[obs_key].cat.categories)
    all_categories = sorted(all_categories)

    distributions = []
    for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key):

        value_counts = adata.obs[obs_key].value_counts(sort=False)
        distribution = np.zeros(len(all_categories))

        for i, category in enumerate(all_categories):
            if category in value_counts:
                distribution[i] = value_counts[category]

        distributions.append(distribution)

    return _jensen_shannon_divergence(np.array(distributions))


def _jensen_shannon_divergence(distributions: np.ndarray) -> float:
    """Compute the Jensen-Shannon divergence (JSD) for a multiple probability distributions.

    The lower the score, the better distribution of clusters among the different batches.

    Args:
        distributions: An array of shape (B, C), where B is the number of batches, and C is the number of clusters. For each batch, it contains the percentage of each cluster among cells.

    Returns:
        A float corresponding to the JSD
    """
    distributions = distributions / distributions.sum(1)[:, None]
    mean_distribution = np.mean(distributions, 0)

    return entropy(mean_distribution) - np.mean([entropy(dist) for dist in distributions])


def entropy(distribution: np.ndarray) -> float:
    """Shannon entropy

    Args:
        distribution: An array of probabilities (should sum to one)

    Returns:
        The Shannon entropy
    """
    return -(distribution * np.log2(distribution + Nums.EPS)).sum()


def mean_normalized_entropy(
    adatas: AnnData | list[AnnData], n_classes: int, obs_key: str, slide_key: str = None
) -> float:
    return np.mean(
        [
            _mean_normalized_entropy(adata, obs_key, n_classes=n_classes)
            for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key)
        ]
    )


def _mean_normalized_entropy(adata: AnnData, obs_key: str, n_classes: int) -> float:
    distribution = adata.obs[obs_key].value_counts(normalize=True).values
    distribution = np.pad(distribution, (0, n_classes - len(distribution)), mode="constant")
    entropy_ = entropy(distribution)

    return entropy_ / np.log2(n_classes)


def heuristic(adata: AnnData | list[AnnData], obs_key: str, n_classes: int, slide_key: str = None) -> float:
    """Heuristic score to evaluate the quality of the clustering.

    Args:
        adata: An `AnnData` object
        obs_key: The key in `adata.obs` that contains the domains.
        n_classes: The number of classes.
        slide_key: The key in `adata.obs` that contains the slide id.

    Returns:
        The heuristic score.
    """
    return np.mean(
        [_heuristic(adata, obs_key, n_classes) for adata in _iter_uid(adata, slide_key=slide_key, obs_key=obs_key)]
    )


def _heuristic(adata: AnnData, obs_key: str, n_classes: int) -> float:
    fide_ = fide_score(adata, obs_key, n_classes=n_classes)

    distribution = adata.obs[obs_key].value_counts(normalize=True).values
    distribution = np.pad(distribution, (0, n_classes - len(distribution)), mode="constant")
    entropy_ = entropy(distribution)

    return fide_ * entropy_ / np.log2(n_classes)


def _iter_uid(adatas: AnnData | list[AnnData], slide_key: str | None = None, obs_key: str | None = None):
    """Iterate over all slides, and make sure `adata.obs[obs_key]` is categorical.

    Args:
        adatas: One or a list of AnnData object(s).
        slide_key: The key in `adata.obs` that contains the slide id.
        obs_key: The key in `adata.obs` that contains the domain id.

    Yields:
        One `AnnData` per slide.
    """
    if isinstance(adatas, AnnData):
        adatas = [adatas]

    if obs_key is not None:
        categories = set.union(*[set(adata.obs[obs_key].astype("category").cat.categories) for adata in adatas])
        for adata in adatas:
            adata.obs[obs_key] = adata.obs[obs_key].astype("category").cat.set_categories(categories)

    for adata in adatas:
        if slide_key is None:
            yield adata
            continue

        for slide_id in adata.obs[slide_key].unique():
            adata_yield = adata[adata.obs[slide_key] == slide_id]

            yield adata_yield
