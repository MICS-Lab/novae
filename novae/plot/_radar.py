import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.patches import Circle
from scipy.cluster.hierarchy import leaves_list, linkage

log = logging.getLogger(__name__)


def orbital(
    adata: AnnData,
    obs_key: str,
    reference_domain: str,
    group_key: str,
    normalize: bool = True,
    figsize: tuple[float, float] = (5, 5),
    max_size: float = 80,
    palette: str = "tab10",
    show: bool = True,
):
    distances = _cells_to_groups(adata, obs_key, reference_domain)
    distances[obs_key] = adata.obs[obs_key]

    distance_per_domain = distances.groupby(obs_key, observed=True)[reference_domain].mean()
    distance_per_domain.loc[reference_domain] = 1

    groups_proportions = (
        adata.obs.groupby(obs_key, observed=True)[group_key].value_counts(normalize=normalize).unstack().T
    )
    groups_proportions = _reorder_df(groups_proportions)

    n_groups = len(groups_proportions)

    groups_angles = np.linspace(0, 2 * np.pi, n_groups + 2)[1:-1]
    domain_outline_width = adata.obs[obs_key].value_counts(normalize=True)

    _, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette(palette, len(distance_per_domain))
    max_distance = distance_per_domain.max()
    width = max_distance * 1.2

    for group, angle in zip(groups_proportions.index, groups_angles):
        for domain, color in zip(distance_per_domain.index, colors):
            radius = distance_per_domain[domain]
            size = groups_proportions.loc[group, domain]
            ax.scatter([radius * np.cos(angle)], [radius * np.sin(angle)], color=color, s=size * max_size)
        plt.annotate(
            group,
            (width * np.cos(angle), width * np.sin(angle)),
            ha="center",
            va="center",
            zorder=2,
        )
        plt.plot(
            [np.cos(angle), max_distance * np.cos(angle)],
            [np.sin(angle), max_distance * np.sin(angle)],
            c="black",
            linewidth=0.5,
            zorder=-1,
        )

    for (domain, radius), color in zip(distance_per_domain.items(), colors):
        circle = Circle((0, 0), radius, label=domain, fill=domain == reference_domain, color=color, alpha=0.5)
        circle.set_linewidth(domain_outline_width[domain] * len(domain_outline_width))
        ax.add_artist(circle)
    ax.legend()

    plt.xlim(-width, width)
    plt.ylim(-width, width)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, frameon=False)
    plt.title(f"Distances of {group_key} and {obs_key} to {reference_domain}")

    ### Figure ticks
    ax.set_yticks([])
    ax.set_xticks(np.arange(1, int(max_distance) + 1))
    ax.xaxis.set_ticks_position("top")
    ax.spines["top"].set_position(("data", 0))
    sns.despine(offset=10, trim=True)
    ax.spines["top"].set_visible(True)
    ax.spines["left"].set_visible(False)

    if show:
        plt.show()


def _cells_to_groups(
    adata: AnnData,
    group_key: str,
    reference_domain: str,
) -> pd.DataFrame | None:
    """Compute the hop-distance between each cell and a cell category/group.

    Args:
        adata: An `AnnData` object, or a `SpatialData object`
        group_key: Key of `adata.obs` containing the groups
        key_added_prefix: Prefix to the key added in `adata.obsm`. If `None`, will return the `DataFrame` instead of saving it.

    Returns:
        A `Dataframe` of shape `n_obs * 1`
    """
    # _check_has_delaunay(adata)

    distances_to_groups = {}

    if not adata.obs[group_key].dtype.name == "category":
        log.info(f"Converting adata.obs['{group_key}'] to category")
        adata.obs[group_key] = adata.obs[group_key].astype("category")

    group_nodes = np.where(adata.obs[group_key] == reference_domain)[0]

    distances = np.full(adata.n_obs, np.nan)

    visited = set()
    queue = group_nodes
    current_distance = 0

    while len(queue):
        distances[queue] = current_distance

        neighbors = set(adata.obsp["spatial_connectivities"][queue].indices)
        queue = np.array(list(neighbors - visited))
        visited |= neighbors

        current_distance += 1

    distances_to_groups[reference_domain] = distances

    return pd.DataFrame(distances_to_groups, index=adata.obs_names)


def _reorder_df(df: pd.DataFrame) -> pd.DataFrame:
    linkage_matrix = linkage(df.values)
    return df.iloc[leaves_list(linkage_matrix)]
