from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix
from torch import Tensor

from .._constants import (
    ADJ,
    COUNTS_LAYER,
    DELAUNAY_RADIUS_TH,
    SLIDE_KEY,
    VAR_MEAN,
    VAR_STD,
)
from ._build import spatial_neighbors

log = logging.getLogger(__name__)


def prepare_adatas(
    adata: AnnData | list[AnnData],
    slide_key: str = None,
) -> list[AnnData]:
    """Ensure the AnnData objects are ready to be used by the model"""
    adatas = [adata] if isinstance(adata, AnnData) else adata

    sanity_check(adatas, slide_key=slide_key)

    for adata in adatas:
        mean = adata.X.mean(0)
        mean = mean.A1 if isinstance(mean, np.matrix) else mean
        adata.var[VAR_MEAN] = mean.astype(np.float32)

        std = adata.X.std(0) if isinstance(adata.X, np.ndarray) else sparse_std(adata.X, 0).A1
        adata.var[VAR_STD] = std.astype(np.float32)

    return adatas


def sanity_check(adatas: list[AnnData], slide_key: str = None):
    """Check that the AnnData objects are preprocessed correctly"""
    count_raw = 0
    count_no_adj = 0

    for adata in adatas:
        if slide_key is not None:
            assert slide_key in adata.obs, f"{slide_key=} must be in all adata.obs"
            values: pd.Series = f"{id(adata)}_" + adata.obs[slide_key].astype(str)
            adata.obs[SLIDE_KEY] = values.astype("category")

        if adata.X.max() >= 10:
            count_raw += 1

            adata.layers[COUNTS_LAYER] = adata.X.copy()
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)

        if ADJ not in adata.obsp:
            spatial_neighbors(adata, radius=[0, DELAUNAY_RADIUS_TH], library_key=slide_key)
            count_no_adj += 1

    if count_no_adj:
        log.warn(
            f"Added delaunay graph to {count_no_adj} adata object(s) with radius threshold {DELAUNAY_RADIUS_TH}"
        )

    if count_raw:
        log.info(
            f"Preprocessed {count_raw} adata object(s) with sc.pp.normalize_total and sc.pp.log1p (raw counts were saved in adata.layers['{COUNTS_LAYER}'])"
        )


def genes_union(adatas: list[AnnData]) -> list[str]:
    return set.union(*[set(adata.var_names) for adata in adatas])


def sparse_std(a: csr_matrix, axis=None) -> np.matrix:
    a_squared = a.multiply(a)
    return np.sqrt(a_squared.mean(axis) - np.square(a.mean(axis)))


def repository_path() -> Path:
    """Get the repository path (dev-mode users only)

    Returns:
        `graph_lr` repository path
    """
    return Path(__file__).parents[2]


def tqdm(*args, desc="DataLoader", **kwargs):
    try:
        import IProgress
        from tqdm.notebook import tqdm
    except ImportError:
        from tqdm import tqdm

    return tqdm(*args, desc=desc, **kwargs)


def fill_invalid_indices(
    out: np.ndarray | Tensor, adata: AnnData, valid_indices: list[int], fillna: int | str = 0
) -> np.ndarray:
    if isinstance(out, Tensor):
        out = out.numpy(force=True)

    dtype = np.float32

    if isinstance(fillna, str):
        out = out.astype(str)
        dtype = "str"

    res = np.full((adata.n_obs, *out.shape[1:]), fillna, dtype=dtype)
    res[valid_indices] = out
    return res
