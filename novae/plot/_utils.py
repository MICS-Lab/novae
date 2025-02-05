import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData

from .._constants import Keys


def get_categorical_color_palette(adatas: list[AnnData], obs_key: str) -> tuple[list[str], list[str]]:
    key_added = f"{obs_key}_colors"

    all_domains = sorted(list(set.union(*[set(adata.obs[obs_key].cat.categories) for adata in adatas])))

    n_colors = len(all_domains)
    colors = list(sns.color_palette("tab10" if n_colors <= 10 else "tab20", n_colors=n_colors).as_hex())
    for adata in adatas:
        adata.obs[obs_key] = adata.obs[obs_key].cat.set_categories(all_domains)
        adata.uns[key_added] = colors

    return all_domains, colors


def _subplots_per_slide(
    adatas: list[AnnData], ncols: int, fig_size_per_slide: tuple[int, int]
) -> tuple[plt.Figure, np.ndarray]:
    n_slides = sum(len(adata.obs[Keys.SLIDE_ID].cat.categories) for adata in adatas)
    ncols = n_slides if n_slides < ncols else ncols
    nrows = (n_slides + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * fig_size_per_slide[0], nrows * fig_size_per_slide[1]), squeeze=False
    )

    return fig, axes


def _get_default_cell_size(adata: AnnData | list[AnnData]) -> float:
    if isinstance(adata, list):
        adata = max(adata, key=lambda adata: adata.n_obs)

    assert (
        Keys.ADJ in adata.obsp
    ), f"Expected {Keys.ADJ} in adata.obsp. Please run `novae.spatial_neighbors(...)` first."

    return np.median(adata.obsp[Keys.ADJ].data)
