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

from .._constants import Keys, Nums
from ._build import spatial_neighbors

log = logging.getLogger(__name__)


def prepare_adatas(
    adata: AnnData | list[AnnData] | None,
    slide_key: str = None,
    var_names: set | list[str] | None = None,
) -> list[AnnData]:
    """Ensure the AnnData objects are ready to be used by the model"""
    assert adata is not None or var_names is not None, "One of `adata` and `var_names` must not be None"

    if adata is None:
        return None, var_names

    adatas = [adata] if isinstance(adata, AnnData) else adata

    sanity_check(adatas, slide_key=slide_key)

    for adata in adatas:
        mean = adata.X.mean(0)
        mean = mean.A1 if isinstance(mean, np.matrix) else mean
        adata.var[Keys.VAR_MEAN] = mean.astype(np.float32)

        std = adata.X.std(0) if isinstance(adata.X, np.ndarray) else sparse_std(adata.X, 0).A1
        adata.var[Keys.VAR_STD] = std.astype(np.float32)

        if var_names is not None and Keys.IS_KNOWN_GENE not in adata.var:
            lookup_valid_genes(adata, var_names)

    if var_names is None:
        var_names = list(genes_union(adatas))

    return adatas, var_names


def keep_highly_variable_genes(adatas: list[AnnData]):
    if max(adata.n_obs for adata in adatas) < Nums.MAX_GENES:
        return

    # TODO: if too many genes, keep only HVGs
    ...


def sanity_check(adatas: list[AnnData], slide_key: str = None):
    """Check that the AnnData objects are preprocessed correctly"""
    count_raw = 0
    count_no_adj = 0

    check_slide_key = True
    if all(Keys.SLIDE_ID in adata.obs for adata in adatas):
        sets = [set(adata.obs[Keys.SLIDE_ID].unique()) for adata in adatas]
        if len(set.union(*sets)) == sum(len(s) for s in sets):
            check_slide_key = False

    for adata in adatas:
        if check_slide_key and slide_key is not None:
            assert slide_key in adata.obs, f"{slide_key=} must be in all adata.obs"
            values: pd.Series = f"{id(adata)}_" + adata.obs[slide_key].astype(str)
            adata.obs[Keys.SLIDE_ID] = values.astype("category")
        elif check_slide_key:
            adata.obs[Keys.SLIDE_ID] = pd.Series(id(adata), index=adata.obs_names, dtype="category")
        else:
            adata.obs[Keys.SLIDE_ID] = adata.obs[Keys.SLIDE_ID].astype("category")

        if adata.X.min() < 0:
            log.warn("Found some negative values in adata.X. It is recommended to have unscaled data (raw or log1p).")

        if adata.X.max() >= 10:
            count_raw += 1

            adata.layers[Keys.COUNTS_LAYER] = adata.X.copy()
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)

        if Keys.ADJ not in adata.obsp:
            spatial_neighbors(adata, radius=[0, Nums.DELAUNAY_RADIUS_TH], library_key=slide_key)
            count_no_adj += 1

    if count_no_adj:
        log.warn(
            f"Added delaunay graph to {count_no_adj} adata object(s) with radius threshold {Nums.DELAUNAY_RADIUS_TH}"
        )

    if count_raw:
        log.info(
            f"Preprocessed {count_raw} adata object(s) with sc.pp.normalize_total and sc.pp.log1p (raw counts were saved in adata.layers['{Keys.COUNTS_LAYER}'])"
        )


def lower_var_names(var_names: pd.Index) -> pd.Index:
    return var_names.str.lower()


def lookup_valid_genes(adata: AnnData, vocabulary: set | list[str]):
    adata.var[Keys.IS_KNOWN_GENE] = np.isin(lower_var_names(adata.var_names), list(vocabulary))

    n_known = sum(adata.var[Keys.IS_KNOWN_GENE])
    assert n_known >= 5, f"Too few genes ({n_known}) are known by the model."
    if n_known / adata.n_vars < 0.50:
        log.warn(f"Only {n_known / adata.n_vars:.1%} of genes are known by the model.")


def genes_union(adatas: list[AnnData]) -> list[str]:
    return set.union(*[set(lower_var_names(adata.var_names)) for adata in adatas])


def sparse_std(a: csr_matrix, axis=None) -> np.matrix:
    a_squared = a.multiply(a)
    return np.sqrt(a_squared.mean(axis) - np.square(a.mean(axis)))


def repository_root() -> Path:
    """Get the path to the root of the repository (dev-mode users only)

    Returns:
        `novae` repository path
    """
    path = Path(__file__).parents[2]

    if path.name != "novae":
        log.warn(f"Trying to get the novae repository path, but it seems it was not installed in dev mode: {path}")

    return path


def wandb_log_dir() -> Path:
    return repository_root() / "wandb"


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
    n_obs: int,
    valid_indices: list[int],
    fill_value: float | str = np.nan,
    dtype: object = None,
) -> np.ndarray:
    if isinstance(out, Tensor):
        out = out.numpy(force=True)

    dtype = np.float32 if dtype is None else dtype

    if isinstance(fill_value, str):
        dtype = object

    res = np.full((n_obs, *out.shape[1:]), fill_value, dtype=dtype)
    res[valid_indices] = out
    return res
