from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from scanpy._utils import sanitize_anndata

from .._constants import Keys

log = logging.getLogger(__name__)


ERROR_ADVICE = "Please run `model.assign_domains(...)` first"


def domains(
    adata: AnnData | list[AnnData],
    k: int | None = None,
    slide_name_key: str | None = None,
    cell_size: int | None = 10,
    ncols: int = 4,
    fig_size_per_slide: tuple[int, int] = (5, 5),
    show: bool = False,
    **kwargs: int,
):
    """Show the Novae spatial domains or niches for all slides in the `AnnData` object.

    Info:
        Make sure you have already your Novae domains assigned to the `AnnData` object. You can use `model.assign_domains(...)` to do so.


    Args:
        adata: An `AnnData` object, or a list of `AnnData` objects.
        k: Number of niches to show (argument from `model.assign_domains(...)`). By default, the last available niche key is shown.
        slide_name_key: Key of `adata.obs` that contains the slide names. By default, uses the Novae unique slide ID.
        cell_size: Size of the cells or spots.
        ncols: Number of columns to be shown.
        fig_size_per_slide: Size of the figure for each slide.
        show: Whether to show the plot.
        **kwargs: Additional arguments for `sc.pl.spatial`.
    """
    adatas = adata if isinstance(adata, list) else [adata]
    slide_name_key = slide_name_key if slide_name_key is not None else Keys.SLIDE_ID

    available_k = _shared_k(adatas)
    if k is None:
        k = list(available_k)[-1]
        obs_key = f"{Keys.NICHE_PREFIX}{k}"
        log.info(f"Showing {obs_key=} as default.")
    else:
        obs_key = f"{Keys.NICHE_PREFIX}{k}"
        assert all(
            obs_key in adata.obs for adata in adatas
        ), f"Novae niches with {k=} not available in all AnnData objects. {ERROR_ADVICE}. Or consider using one of {available_k} instead."

    for adata in adatas:
        sanitize_anndata(adata)

    all_domains = sorted(list(set.union(*[set(adata.obs[obs_key].cat.categories) for adata in adatas])))

    n_colors = len(all_domains)
    colors = list(sns.color_palette("tab10" if n_colors <= 10 else "tab20", n_colors=n_colors).as_hex())
    for adata in adatas:
        adata.obs[obs_key] = adata.obs[obs_key].cat.set_categories(all_domains)
        if len(all_domains) <= 10:
            adata.uns[f"{obs_key}_colors"] = colors

    n_slides = sum(len(adata.obs[slide_name_key].cat.categories) for adata in adatas)
    ncols = n_slides if n_slides < ncols else ncols
    nrows = (n_slides + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * fig_size_per_slide[0], nrows * fig_size_per_slide[1]), squeeze=False
    )

    i = 0
    for adata in adatas:
        for sid in adata.obs[slide_name_key].cat.categories:
            ax = axes[i // ncols, i % ncols]
            adata_ = adata[adata.obs[slide_name_key] == sid]
            sc.pl.spatial(adata_, spot_size=cell_size, color=obs_key, ax=ax, show=False, **kwargs)
            sns.despine(ax=ax, offset=10, trim=True)
            ax.set_title(sid)
            i += 1

    [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]
    fig.suptitle(f"Novae domains ({k=})", fontsize=14)

    if show:
        plt.show()


def _available_k(adata: AnnData) -> set[int]:
    return {
        int(key[len(Keys.NICHE_PREFIX) :])
        for key in adata.obs.columns[adata.obs.columns.str.startswith(Keys.NICHE_PREFIX)]
    }


def _shared_k(adatas: list[AnnData]) -> set[int]:
    available_ks = [_available_k(adata) for adata in adatas]
    assert any(available_ks), f"No Novae niches available. {ERROR_ADVICE}"

    available_k = set.intersection(*available_ks)
    assert available_k, f"No common Novae niches available. {ERROR_ADVICE}"

    return available_k
