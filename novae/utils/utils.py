from __future__ import annotations

import importlib
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
    ADJ_LOCAL,
    COUNTS_LAYER,
    DELAUNAY_RADIUS_TH,
    IS_KNOWN_GENE_KEY,
    SLIDE_KEY,
    VAR_MEAN,
    VAR_STD,
)
from ._build import spatial_neighbors

log = logging.getLogger(__name__)


def prepare_adatas(
    adata: AnnData | list[AnnData],
    slide_key: str = None,
    vocabulary: set | None = None,
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

        if vocabulary is not None and IS_KNOWN_GENE_KEY not in adata.var:
            lookup_valid_genes(adata, vocabulary)

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
        else:
            adata.obs[SLIDE_KEY] = pd.Series(id(adata), index=adata.obs_names, dtype="category")

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


def lower_var_names(var_names: pd.Index) -> pd.Index:
    return var_names.map(lambda s: s.lower())


def lookup_valid_genes(adata: AnnData, vocabulary: set):
    adata.var[IS_KNOWN_GENE_KEY] = np.isin(lower_var_names(adata.var_names), list(vocabulary))

    n_known = sum(adata.var[IS_KNOWN_GENE_KEY])
    assert n_known >= 5, f"Too few genes ({n_known}) are known by the model."
    if n_known / adata.n_vars < 0.50:
        log.warn(f"Only {n_known / adata.n_vars:.1%} of genes are known by the model.")


def genes_union(adatas: list[AnnData]) -> list[str]:
    return set.union(*[set(lower_var_names(adata.var_names)) for adata in adatas])


def sparse_std(a: csr_matrix, axis=None) -> np.matrix:
    a_squared = a.multiply(a)
    return np.sqrt(a_squared.mean(axis) - np.square(a.mean(axis)))


def repository_path() -> Path:
    """Get the repository path (dev-mode users only)

    Returns:
        `novae` repository path
    """
    return Path(__file__).parents[2]


def tqdm(*args, desc="DataLoader", **kwargs):
    # check if ipywidgets is installed before importing tqdm.auto
    # to ensure it won't fail and a progress bar is displayed
    if importlib.util.find_spec("ipywidgets") is not None:
        from tqdm.auto import tqdm as _tqdm
    else:
        from tqdm import tqdm as _tqdm

    return _tqdm(*args, desc=desc, **kwargs)


def fill_invalid_indices(
    out: np.ndarray | Tensor,
    adata: AnnData,
    valid_indices: list[int],
    fill_value: float | str = 0,
    dtype: object = None,
) -> np.ndarray:
    if isinstance(out, Tensor):
        out = out.numpy(force=True)

    dtype = np.float32 if dtype is None else dtype

    if isinstance(fill_value, str):
        dtype = object

    res = np.full((adata.n_obs, *out.shape[1:]), fill_value, dtype=dtype)
    res[valid_indices] = out
    return res


def fill_edge_scores(
    out: np.ndarray | Tensor,
    adata: AnnData,
    valid_indices: list[int],
    fill_value: float | str = 0,
    dtype: object = None,
) -> np.ndarray:
    """
    TODO
    """
    for index in valid_indices[0]:
        indices = adata.obsp[ADJ_LOCAL][index].indices
        adjacency = adata.obsp[ADJ]
        adjacency_scores = adjacency.copy()
    pass