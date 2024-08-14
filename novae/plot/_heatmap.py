import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData

from .._constants import Keys


def _weights_clustermap(
    weights: np.ndarray,
    adatas: list[AnnData] | None,
    slide_ids: list[str],
    show_yticklabels: bool = False,
    show_tissue_legend: bool = True,
    figsize: tuple[int] = (6, 4),
    vmin: float = 0.9,
    vmax: float = 1.1,
    **kwargs: int,
) -> None:
    row_colors = None
    if show_tissue_legend and adatas is not None and all(Keys.UNS_TISSUE in adata.uns for adata in adatas):
        tissues = list({adata.uns[Keys.UNS_TISSUE] for adata in adatas})
        tissue_colors = {tissue: sns.color_palette("tab20")[i] for i, tissue in enumerate(tissues)}

        row_colors = []
        for slide_id in slide_ids:
            for adata in adatas:
                if adata.obs[Keys.SLIDE_ID].iloc[0] == slide_id:
                    row_colors.append(tissue_colors[adata.uns[Keys.UNS_TISSUE]])
                    break
            else:
                row_colors.append("gray")

    sns.clustermap(
        weights,
        yticklabels=slide_ids if show_yticklabels else False,
        xticklabels=False,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        row_colors=row_colors,
        **kwargs,
    )

    if row_colors is not None and show_tissue_legend:
        handles = [mpatches.Patch(facecolor=color, label=tissue) for tissue, color in tissue_colors.items()]
        ax = plt.gcf().axes[3]
        ax.legend(handles=handles, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, frameon=False)
