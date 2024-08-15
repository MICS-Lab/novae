from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def _leaves_count(clustering: AgglomerativeClustering) -> np.ndarray:
    counts = np.zeros(clustering.children_.shape[0])
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    return counts


def _domains_hierarchy(
    clustering: AgglomerativeClustering,
    max_level: int = 10,
    hline_level: int | list[int] | None = None,
    leaf_font_size: int = 10,
    **kwargs,
) -> None:
    assert max_level > 1

    size = clustering.children_.shape[0]
    original_ymax = max_level + 1
    original_ticks = np.arange(1, original_ymax)
    height = original_ymax + np.arange(size) - size

    if hline_level is not None:
        hline_level = [hline_level] if isinstance(hline_level, int) else hline_level
        for level in hline_level:
            plt.hlines(original_ymax - hline_level, 0, 1e5, colors="r", linestyles="dashed")

    linkage_matrix = np.column_stack([clustering.children_, height.clip(0), _leaves_count(clustering)]).astype(float)

    ddata = dendrogram(
        linkage_matrix,
        color_threshold=-1,
        leaf_font_size=leaf_font_size,
        p=max_level + 1,
        truncate_mode="lastp",
        above_threshold_color="#ccc",
        **kwargs,
    )

    for i, d in zip(ddata["icoord"][::-1], ddata["dcoord"][::-1]):
        x, y = 0.5 * sum(i[1:3]), d[1]
        plt.plot(x, y, "ko")
        plt.annotate(f"N{size - 1 + int(y)}", (x, y), xytext=(0, -8), textcoords="offset points", va="top", ha="center")

    plt.yticks(original_ticks, original_ymax - original_ticks)

    plt.xlabel(None)
    plt.ylabel("Domains level")
    plt.title("Domains hierarchy")
    plt.xlabel("Number of domains in node (or prototype ID if no parenthesis)")
    sns.despine(offset=10, trim=True, bottom=True)
