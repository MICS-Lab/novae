from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData

from .. import utils
from .._constants import Keys
from ._utils import get_categorical_color_palette


def domains_proportions(adata: AnnData | list[AnnData], obs_key: str | None, figsize: tuple[int, int] = (2, 5)):
    adatas = [adata] if isinstance(adata, AnnData) else adata

    obs_key = utils.check_available_domains_key(adatas, obs_key)

    all_domains, colors = get_categorical_color_palette(adatas, obs_key)

    names, series = [], []
    for adata_slide in utils.iter_slides(adatas):
        names.append(adata_slide.obs[Keys.SLIDE_ID].iloc[0])
        series.append(adata_slide.obs[obs_key].value_counts(normalize=True))

    df = pd.concat(series, axis=1)
    df.columns = names

    df.T.plot(kind="bar", stacked=True, figsize=figsize, color=dict(zip(all_domains, colors)))
    sns.despine(offset=10, trim=True)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, frameon=False)
    plt.ylabel("Proportion")
