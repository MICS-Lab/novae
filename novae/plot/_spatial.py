from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.lines import Line2D
from scanpy._utils import sanitize_anndata

from .. import utils
from .._constants import Keys

log = logging.getLogger(__name__)


def domains(
    adata: AnnData | list[AnnData],
    obs_key: str | None = None,
    slide_name_key: str | None = None,
    cell_size: int | None = 10,
    ncols: int = 4,
    fig_size_per_slide: tuple[int, int] = (5, 5),
    na_color: str = "#ccc",
    show: bool = False,
    **kwargs: int,
):
    """Show the Novae spatial domains for all slides in the `AnnData` object.

    Info:
        Make sure you have already your Novae domains assigned to the `AnnData` object. You can use `model.assign_domains(...)` to do so.


    Args:
        adata: An `AnnData` object, or a list of `AnnData` objects.
        obs_key: Name of the key from `adata.obs` containing the Novae domains. By default, the last available domain key is shown.
        slide_name_key: Key of `adata.obs` that contains the slide names. By default, uses the Novae unique slide ID.
        cell_size: Size of the cells or spots.
        ncols: Number of columns to be shown.
        fig_size_per_slide: Size of the figure for each slide.
        na_color: Color for cells that does not belong to any domain (i.e. cells with a too small neighborhood).
        show: Whether to show the plot.
        **kwargs: Additional arguments for `sc.pl.spatial`.
    """
    if obs_key is not None:
        assert str(obs_key).startswith(Keys.DOMAINS_PREFIX), f"Received {obs_key=}, which is not a valid Novae obs_key"

    adatas = adata if isinstance(adata, list) else [adata]
    slide_name_key = slide_name_key if slide_name_key is not None else Keys.SLIDE_ID
    obs_key = utils.check_available_domains_key(adatas, obs_key)

    for adata in adatas:
        sanitize_anndata(adata)

    all_domains = sorted(list(set.union(*[set(adata.obs[obs_key].cat.categories) for adata in adatas])))

    n_colors = len(all_domains)
    colors = list(sns.color_palette("tab10" if n_colors <= 10 else "tab20", n_colors=n_colors).as_hex())
    for adata in adatas:
        adata.obs[obs_key] = adata.obs[obs_key].cat.set_categories(all_domains)
        adata.uns[f"{obs_key}_colors"] = colors

    n_slides = sum(len(adata.obs[slide_name_key].cat.categories) for adata in adatas)
    ncols = n_slides if n_slides < ncols else ncols
    nrows = (n_slides + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * fig_size_per_slide[0], nrows * fig_size_per_slide[1]), squeeze=False
    )

    i = 0
    for adata in adatas:
        for slide_name in adata.obs[slide_name_key].cat.categories:
            ax = axes[i // ncols, i % ncols]
            adata_ = adata[adata.obs[slide_name_key] == slide_name]
            sc.pl.spatial(adata_, spot_size=cell_size, color=obs_key, ax=ax, show=False, **kwargs)
            sns.despine(ax=ax, offset=10, trim=True)
            ax.get_legend().remove()
            ax.set_title(slide_name)
            i += 1

    [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]  # remove unused subplots

    title = f"Novae domains ({obs_key})"

    if i == 1:
        axes[0, 0].set_title(title)
    else:
        fig.suptitle(title, fontsize=14, y=1.15)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, linestyle="None")
        for color in colors + [na_color]
    ]
    fig.legend(
        handles,
        all_domains + ["NA"],
        loc="upper center" if i > 1 else "center left",
        bbox_to_anchor=(0.5, 1.1) if i > 1 else (1.04, 0.5),
        borderaxespad=0,
        frameon=False,
        ncol=n_colors // (3 if i > 1 else 10) + 1,
    )

    if show:
        plt.show()
