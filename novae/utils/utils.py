from __future__ import annotations

import importlib
import logging
from pathlib import Path

import anndata
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

    _sanity_check(adatas, slide_key=slide_key)

    _lookup_highly_variable_genes(adatas)

    for adata in adatas:
        mean = adata.X.mean(0)
        mean = mean.A1 if isinstance(mean, np.matrix) else mean
        adata.var[Keys.VAR_MEAN] = mean.astype(np.float32)

        std = adata.X.std(0) if isinstance(adata.X, np.ndarray) else _sparse_std(adata.X, 0).A1
        adata.var[Keys.VAR_STD] = std.astype(np.float32)

        _lookup_valid_genes(adata, var_names)

        adata.var[Keys.USE_GENE] = _var_or_true(adata, Keys.HIGHLY_VARIABLE) & _var_or_true(adata, Keys.IS_KNOWN_GENE)

        n_used = adata.var[Keys.USE_GENE].sum()
        assert (
            n_used >= Nums.MIN_GENES
        ), f"Too few genes ({n_used}) are both (i) known by the model and (ii) highly variable."

    if var_names is None:
        var_names = list(genes_union(adatas, among_used=True))

    return adatas, var_names


def _var_or_true(adata: AnnData, key: str) -> pd.Series | bool:
    return adata.var[key] if key in adata.var else True


def _lookup_highly_variable_genes(adatas: list[AnnData]):
    if max(adata.n_vars for adata in adatas) <= Nums.MAX_GENES:
        return

    if len(adatas) == 0 or _is_multi_panel(adatas):
        for adata in adatas:
            sc.pp.highly_variable_genes(adata)
        return

    adata_ = anndata.concat(adatas)

    assert adata_.n_vars == adatas[0].n_vars, "Same panel used but the gene names don't have the same case"

    sc.pp.highly_variable_genes(adata_)
    highly_variable_genes = adata_.var_names[adata_.var[Keys.HIGHLY_VARIABLE]]

    for adata in adatas:
        adata.var[Keys.HIGHLY_VARIABLE] = False
        adata.var.loc[highly_variable_genes, Keys.HIGHLY_VARIABLE] = True


def _is_multi_panel(adatas: list[AnnData]) -> bool:
    if len(adatas) == 1:
        return False

    first_panel = sorted(lower_var_names(adatas[0].var_names))

    for adata in adatas[1:]:
        if sorted(lower_var_names(adata.var_names)) != first_panel:
            return True

    return False


def _sanity_check(adatas: list[AnnData], slide_key: str = None):
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
            f"Added delaunay graph to {count_no_adj} adata object(s) with default radius threshold ({Nums.DELAUNAY_RADIUS_TH} microns)"
        )

    if count_raw:
        log.info(
            f"Preprocessed {count_raw} adata object(s) with sc.pp.normalize_total and sc.pp.log1p (raw counts were saved in adata.layers['{Keys.COUNTS_LAYER}'])"
        )


def lower_var_names(var_names: pd.Index | list[str]) -> pd.Index | list[str]:
    if isinstance(var_names, pd.Index):
        return var_names.str.lower()
    return [name.lower() for name in var_names]


def _lookup_valid_genes(adata: AnnData, var_names: set | list[str] | None):
    if var_names is None or Keys.IS_KNOWN_GENE in adata.var:
        return

    adata.var[Keys.IS_KNOWN_GENE] = np.isin(lower_var_names(adata.var_names), list(var_names))

    n_known = sum(adata.var[Keys.IS_KNOWN_GENE])
    assert n_known >= Nums.MIN_GENES, f"Too few genes ({n_known}) are known by the model."
    if n_known / adata.n_vars < 0.50:
        log.warn(f"Only {n_known / adata.n_vars:.1%} of genes are known by the model.")


def genes_union(adatas: list[AnnData], among_used: bool = False) -> list[str]:
    if among_used:
        var_names_list = [adata.var_names[adata.var[Keys.USE_GENE]] for adata in adatas]
    else:
        var_names_list = [adata.var_names for adata in adatas]

    return set.union(*[set(lower_var_names(var_names)) for var_names in var_names_list])


def _sparse_std(a: csr_matrix, axis=None) -> np.matrix:
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
