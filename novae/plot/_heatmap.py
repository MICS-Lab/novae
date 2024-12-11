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

    if show_tissue_legend and row_colors is not None:
        handles = [mpatches.Patch(facecolor=color, label=tissue) for tissue, color in tissue_colors.items()]
        ax = plt.gcf().axes[3]
        ax.legend(handles=handles, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, frameon=False)


TEMP_KEY = "_temp"


def pathway_scores(
    adata: AnnData,
    pathways: dict[str, list[str]] | str,
    obs_key: str | None = None,
    pathway_name: str | None = None,
    slide_name_key: str | None = None,
    return_df: bool = False,
    figsize: tuple[int, int] = (10, 5),
    min_pathway_size: int = 4,
    show: bool = True,
    **kwargs: int,
) -> pd.DataFrame | None:
    """Show a heatmap of either (i) the score of multiple pathways for each domain, or (ii) one pathway score for each domain and for each slide.
    To use the latter case, provide `pathway_name`, or make sure to have only one pathway in `pathways`.

    Info:
        Currently, this function only supports one AnnData object per call.

    Args:
        adata: An `AnnData` object.
        pathways: Either a dictionary of pathways (keys are pathway names, values are lists of gene names), or a path to a [GSEA](https://www.gsea-msigdb.org/gsea/msigdb/index.jsp) JSON file.
        obs_key: Key in `adata.obs` that contains the domains. By default, it will use the last available Novae domain key.
        pathway_name: If `None`, all pathways will be shown (first mode). If not `None`, this specific pathway will be shown, for all domains and all slides (second mode).
        slide_name_key: Key of `adata.obs` that contains the slide names. By default, uses the Novae unique slide ID.
        return_df: Whether to return the DataFrame.
        figsize: Matplotlib figure size.
        min_pathway_size: Minimum number of known genes in the pathway to be considered.
        show: Whether to show the plot.

    Returns:
        A DataFrame of scores per domain if `return_df` is True.
    """
    assert isinstance(adata, AnnData), f"For now, only one AnnData object is supported, received {type(adata)}"

    obs_key = utils.check_available_domains_key([adata], obs_key)

    if isinstance(pathways, str):
        pathways = _load_gsea_json(pathways)
        log.info(f"Loaded {len(pathways)} pathway(s)")

    if len(pathways) == 1:
        pathway_name = list(pathways.keys())[0]

    if pathway_name is not None:
        gene_names = pathways[pathway_name]
        is_valid = _get_pathway_score(adata, gene_names, min_pathway_size)
        assert is_valid, f"Pathway '{pathway_name}' has less than {min_pathway_size} genes in the dataset."
    else:
        scores = {}

        for key, gene_names in pathways.items():
            is_valid = _get_pathway_score(adata, gene_names, min_pathway_size)
            if is_valid:
                scores[key] = adata.obs[TEMP_KEY]

    if pathway_name is not None:
        log.info(f"Plot mode: {pathway_name} score per domain per slide")

        slide_name_key = utils.check_slide_name_key(adata, slide_name_key)

        df = adata.obs.groupby([obs_key, slide_name_key], observed=True)[TEMP_KEY].mean().unstack()
        df.columns.name = slide_name_key

        assert len(df) > 1, f"Found {len(df)} valid slide. Minimum 2 required."
    else:
        log.info(f"Plot mode: {len(scores)} pathways scores per domain")

        assert len(scores) > 1, f"Found {len(scores)} valid pathway. Minimum 2 required."

        df = pd.DataFrame(scores)
        df[obs_key] = adata.obs[obs_key]
        df = df.groupby(obs_key, observed=True).mean()
        df.columns.name = "Pathways"

    del adata.obs[TEMP_KEY]

    df = df.fillna(0)

    g = sns.clustermap(df, figsize=figsize, **kwargs)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    if show:
        plt.show()

    if return_df:
        return df


def _get_pathway_score(adata: AnnData, gene_names: list[str], min_pathway_size: int) -> bool:
    lower_var_names = adata.var_names.str.lower()

    vars = np.array([gene_name.lower() for gene_name in gene_names])
    vars = adata.var_names[np.isin(lower_var_names, vars)]

    if len(vars) >= min_pathway_size:
        sc.tl.score_genes(adata, vars, score_name=TEMP_KEY)
        return True
    return False


def _load_gsea_json(path: str) -> dict[str, list[str]]:
    with open(path, "r") as f:
        content: dict = json.load(f)
        assert all(
            "geneSymbols" in value for value in content.values()
        ), "Missing 'geneSymbols' key in JSON file. Expected a valid GSEA JSON file."
        return {key: value["geneSymbols"] for key, value in content.items()}
