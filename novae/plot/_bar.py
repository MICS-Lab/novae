import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData

from .. import utils
from ._utils import get_categorical_color_palette


def domains_proportions(
    adata: AnnData | list[AnnData],
    obs_key: str | None = None,
    slide_name_key: str | None = None,
    figsize: tuple[int, int] = (2, 5),
    show: bool = True,
):
    """Show the proportion of each domain in the slide(s).

    Args:
        adata: One `AnnData` object, or a list of `AnnData` objects.
        obs_key: The key in `adata.obs` that contains the Novae domains. By default, the last available domain key is shown.
        figsize: Matplotlib figure size.
        show: Whether to show the plot.
    """
    adatas = [adata] if isinstance(adata, AnnData) else adata
    slide_name_key = utils.check_slide_name_key(adatas, slide_name_key)
    obs_key = utils.check_available_domains_key(adatas, obs_key)

    all_domains, colors = get_categorical_color_palette(adatas, obs_key)

    names, series = [], []
    for adata_slide in utils.iter_slides(adatas):
        names.append(adata_slide.obs[slide_name_key].iloc[0])
        series.append(adata_slide.obs[obs_key].value_counts(normalize=True))

    df = pd.concat(series, axis=1)
    df.columns = names

    df.T.plot(kind="bar", stacked=True, figsize=figsize, color=dict(zip(all_domains, colors)))
    sns.despine(offset=10, trim=True)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, frameon=False)
    plt.ylabel("Proportion")
    plt.xticks(rotation=90)

    if show:
        plt.show()
