import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.lines import Line2D
from scanpy._utils import sanitize_anndata

from .. import utils
from .._constants import Keys
from ._utils import (
    _get_default_cell_size,
    _subplots_per_slide,
    get_categorical_color_palette,
)

log = logging.getLogger(__name__)


def domains(
    adata: AnnData | list[AnnData],
    obs_key: str | None = None,
    slide_name_key: str | None = None,
    cell_size: int | None = None,
    ncols: int = 4,
    fig_size_per_slide: tuple[int, int] = (5, 5),
    na_color: str = "#ccc",
    show: bool = True,
    library_id: str | None = None,
    **kwargs: int,
):
    """Show the Novae spatial domains for all slides in the `AnnData` object.

    Info:
        Make sure you have already your Novae domains assigned to the `AnnData` object. You can use `model.assign_domains(...)` to do so.

    Args:
        adata: An `AnnData` object, or a list of `AnnData` objects.
        obs_key: Name of the key from `adata.obs` containing the Novae domains. By default, the last available domain key is shown.
        slide_name_key: Key of `adata.obs` that contains the slide names. By default, uses the Novae unique slide ID.
        cell_size: Size of the cells or spots. By default, it uses the median distance between neighbor cells.
        ncols: Number of columns to be shown.
        fig_size_per_slide: Size of the figure for each slide.
        na_color: Color for cells that does not belong to any domain (i.e. cells with a too small neighborhood).
        show: Whether to show the plot.
        library_id: `library_id` argument for `sc.pl.spatial`.
        **kwargs: Additional arguments for `sc.pl.spatial`.
    """
    if obs_key is not None:
        assert str(obs_key).startswith(Keys.DOMAINS_PREFIX), f"Received {obs_key=}, which is not a valid Novae obs_key"

    adatas = adata if isinstance(adata, list) else [adata]
    slide_name_key = utils.check_slide_name_key(adatas, slide_name_key)
    obs_key = utils.check_available_domains_key(adatas, obs_key)

    for adata in adatas:
        sanitize_anndata(adata)

    all_domains, colors = get_categorical_color_palette(adatas, obs_key)
    cell_size = cell_size or _get_default_cell_size(adata)

    fig, axes = _subplots_per_slide(adatas, ncols, fig_size_per_slide)

    for i, adata in enumerate(utils.iter_slides(adatas)):
        ax = axes[i // ncols, i % ncols]
        slide_name = adata.obs[slide_name_key].iloc[0]
        assert len(np.unique(adata.obs[slide_name_key])) == 1

        sc.pl.spatial(
            adata,
            spot_size=cell_size,
            color=obs_key,
            ax=ax,
            show=False,
            library_id=library_id,
            **kwargs,
        )
        sns.despine(ax=ax, offset=10, trim=True)
        ax.get_legend().remove()
        ax.set_title(slide_name)

    [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]  # remove unused subplots

    title = f"Novae domains ({obs_key})"

    if i == 0:
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
        ncol=len(colors) // (3 if i > 1 else 10) + 1,
    )

    if show:
        plt.show()


def spatially_variable_genes(
    adata: AnnData,
    obs_key: str | None = None,
    top_k: int = 5,
    cell_size: int | None = None,
    min_positive_ratio: float = 0.05,
    return_list: bool = False,
    show: bool = True,
    **kwargs: int,
) -> None | list[str]:
    """Plot the most spatially variable genes (SVG) for a given `AnnData` object.

    !!! info
        Currently, this function only supports one slide per call.

    Args:
        adata: An `AnnData` object corresponding to one slide.
        obs_key: Key in `adata.obs` that contains the domains. By default, it will use the last available Novae domain key.
        top_k: Number of SVG to be shown.
        cell_size: Size of the cells or spots (`spot_size` argument of `sc.pl.spatial`). By default, it uses the median distance between neighbor cells.
        min_positive_ratio: Genes whose "ratio of cells expressing it" is lower than this threshold are not considered.
        return_list: Whether to return the list of SVG instead of plotting them.
        show: Whether to show the plot.
        **kwargs: Additional arguments for `sc.pl.spatial`.

    Returns:
        A list of SVG names if `return_list` is `True`.
    """
    assert isinstance(adata, AnnData), f"Received adata of type {type(adata)}. Currently only AnnData is supported."

    obs_key = utils.check_available_domains_key([adata], obs_key)

    sc.tl.rank_genes_groups(adata, groupby=obs_key)
    df = pd.concat(
        [
            sc.get.rank_genes_groups_df(adata, domain).set_index("names")["logfoldchanges"]
            for domain in adata.obs[obs_key].cat.categories
        ],
        axis=1,
    )

    where = (adata.X > 0).mean(0) > min_positive_ratio
    valid_vars = adata.var_names[where.A1 if isinstance(where, np.matrix) else where]
    assert (
        len(valid_vars) >= top_k
    ), f"Only {len(valid_vars)} genes are available. Please decrease `top_k` or `min_positive_ratio`."

    svg = df.std(1).loc[valid_vars].sort_values(ascending=False).head(top_k).index

    if return_list:
        return svg.tolist()

    sc.pl.spatial(adata, color=svg, spot_size=cell_size or _get_default_cell_size(adata), show=show, **kwargs)
