import numpy as np
import pandas as pd
from anndata import AnnData

from .._constants import Keys


def _slides_indices(adatas: list[AnnData], only_valid_obs: bool = True) -> tuple[list[int], list[np.ndarray]]:
    adata_indices, slides_obs_indices = [], []

    for i, adata in enumerate(adatas):
        slide_ids = adata.obs[Keys.SLIDE_ID].cat.categories

        for slide_id in slide_ids:
            condition = adata.obs[Keys.SLIDE_ID] == slide_id
            if only_valid_obs:
                condition = condition & adata.obs[Keys.IS_VALID_OBS]

            adata_indices.append(i)
            slides_obs_indices.append(np.where(condition)[0])

    return adata_indices, slides_obs_indices


def _domains_counts(adata: AnnData, i: int, obs_key: str) -> pd.DataFrame:
    df = adata.obs[[Keys.SLIDE_ID, obs_key]].groupby(Keys.SLIDE_ID, observed=False)[obs_key].value_counts().unstack()
    df[Keys.ADATA_INDEX] = i
    return df


def _domains_counts_per_slide(adatas: list[AnnData], obs_key: str) -> pd.DataFrame:
    return pd.concat([_domains_counts(adata, i, obs_key) for i, adata in enumerate(adatas)], axis=0)


def batch_effect_correction(adatas: list[AnnData], obs_key: str) -> None:
    for adata in adatas:
        assert obs_key in adata.obs, f"Did not found `adata.obs['{obs_key}']`"
        assert (
            Keys.REPR in adata.obsm
        ), f"Did not found `adata.obsm['{Keys.REPR}']`. Please run `model.compute_representations(...)` first"

    adata_indices, slides_obs_indices = _slides_indices(adatas)

    domains_counts_per_slide = _domains_counts_per_slide(adatas, obs_key)
    domains = domains_counts_per_slide.columns[:-1]
    ref_slide_ids: pd.Series = domains_counts_per_slide[domains].idxmax(axis=0)

    def _centroid_reference(domain: str, slide_id: str, obs_key: str):
        adata_ref_index: int = domains_counts_per_slide[Keys.ADATA_INDEX].loc[slide_id]
        adata_ref = adatas[adata_ref_index]
        where = (adata_ref.obs[Keys.SLIDE_ID] == slide_id) & (adata_ref.obs[obs_key] == domain)
        return adata_ref.obsm[Keys.REPR][where].mean(0)

    centroids_reference = pd.DataFrame(
        {domain: _centroid_reference(domain, slide_id, obs_key) for domain, slide_id in ref_slide_ids.items()}
    )

    for adata in adatas:
        adata.obsm[Keys.REPR_CORRECTED] = adata.obsm[Keys.REPR].copy()

    for adata_index, obs_indices in zip(adata_indices, slides_obs_indices):
        adata = adatas[adata_index]

        for domain in domains:
            if adata.obs[Keys.SLIDE_ID].iloc[obs_indices[0]] == ref_slide_ids.loc[domain]:
                continue  # reference for this domain

            indices_domain = obs_indices[adata.obs.iloc[obs_indices][obs_key] == domain]
            if len(indices_domain) == 0:
                continue

            centroid_reference = centroids_reference[domain].values
            centroid = adata.obsm[Keys.REPR][indices_domain].mean(0)

            adata.obsm[Keys.REPR_CORRECTED][indices_domain] += centroid_reference - centroid
