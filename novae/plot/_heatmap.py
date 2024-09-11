from __future__ import annotations

import json
import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData

from .. import utils
from .._constants import Keys

log = logging.getLogger(__name__)


def _weights_clustermap(
    weights: np.ndarray,
    adatas: list[AnnData] | None,
    slide_ids: list[str],
    show_yticklabels: bool = False,
    show_tissue_legend: bool = True,
    figsize: tuple[int] = (6, 4),
    vmin: float = 0,
    vmax: float = 1,
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
        log.info(f"Using {row_colors=}")

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


def pathway_scores(
    adata: AnnData,
    pathways: dict[str, list[str]] | str,
    obs_key: str | None = None,
    return_df: bool = False,
    figsize: tuple[int, int] = (10, 5),
    min_pathway_size: int = 4,
    **kwargs: int,
) -> pd.DataFrame | None:
    """Show a heatmap of pathway scores for each domain.

    Args:
        adata: An `AnnData` object.
        pathways: Either a dictionary of pathways (keys are pathway names, values are lists of gane names), or a path to a [GSEA](https://www.gsea-msigdb.org/gsea/msigdb/index.jsp) JSON file.
        obs_key: Key in `adata.obs` that contains the domains. By default, it will use the last available Novae domain key.
        return_df: Whether to return the DataFrame.
        figsize: Matplotlib figure size.
        min_pathway_size: Minimum number of known genes in the pathway to be considered.

    Returns:
        A DataFrame of scores per domain if `return_df` is True.
    """
    assert isinstance(adata, AnnData), f"For now, only AnnData objects are supported, received {type(adata)}"

    obs_key = utils.check_available_domains_key([adata], obs_key)

    scores = {}
    lower_var_names = adata.var_names.str.lower()

    if isinstance(pathways, str):
        pathways = _load_gsea_json(pathways)
        log.info(f"Loaded {len(pathways)} pathway(s)")

    for key, gene_names in pathways.items():
        vars = np.array([gene_name.lower() for gene_name in gene_names])
        vars = adata.var_names[np.isin(lower_var_names, vars)]
        if len(vars) >= min_pathway_size:
            sc.tl.score_genes(adata, vars, score_name="_temp")
            scores[key] = adata.obs["_temp"]
    del adata.obs["_temp"]

    assert len(scores) > 1, f"Found {len(scores)} valid pathway. Minimum 2 required."

    df = pd.DataFrame(scores)
    df[obs_key] = adata.obs[obs_key]
    df = df.groupby(obs_key).mean()
    df = df.fillna(0)

    g = sns.clustermap(df, figsize=figsize, **kwargs)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    if return_df:
        return df


def _load_gsea_json(path: str) -> dict[str, list[str]]:
    with open(path, "r") as f:
        content: dict = json.load(f)
        assert all(
            "geneSymbols" in value for value in content.values()
        ), "Missing 'geneSymbols' key in JSON file. Expected a valid GSEA JSON file."
        return {key: value["geneSymbols"] for key, value in content.items()}
