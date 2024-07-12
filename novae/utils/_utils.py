from __future__ import annotations

import importlib
import logging
from functools import wraps
from pathlib import Path
from typing import Callable

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,
)
from scipy.sparse import csr_matrix
from torch import Tensor

from .._constants import Keys, Nums
from . import format_docs, spatial_neighbors

log = logging.getLogger(__name__)


@format_docs
def prepare_adatas(
    adata: AnnData | list[AnnData] | None,
    slide_key: str = None,
    var_names: set | list[str] | None = None,
) -> list[AnnData]:
    """Ensure the AnnData objects are ready to be used by the model.

    Note:
        It performs the following operations:

        - Preprocess the data if needed (e.g. normalize, log1p), in which case raw counts are saved in `adata.layers['counts']`
        - Compute spatial neighbors (if not already computed), using [novae.utils.spatial_neighbors][]
        - Compute the mean and std of each gene
        - Save which genes are highly variable, in case the number of genes is too high
        - If `slide_key` is provided, ensure that all `slide_key` are valid and unique
        - If using a pretrained model, save which genes are known by the model


    Args:
        {adata}
        {slide_key}
        {var_names}

    Returns:
        A list of `AnnData` objects ready to be used by the model. If only one `adata` object is provided, it will be wrapped in a list.
    """
    assert adata is not None or var_names is not None, "One of `adata` and `var_names` must not be None"

    if adata is None:
        return None, var_names

    adatas = [adata] if isinstance(adata, AnnData) else adata

    assert len(adatas) > 0, "No `adata` object found. Please provide an AnnData object, or a list of AnnData objects."

    _sanity_check(adatas, slide_key=slide_key)

    _lookup_highly_variable_genes(adatas)

    for adata in adatas:
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


def requires_fit(f: Callable) -> Callable:
    """Make sure the model has been trained"""

    @wraps(f)
    def wrapper(model, *args, **kwargs):
        assert model._trained, "Novae must be trained first, so consider running `model.fit()`"
        return f(model, *args, **kwargs)

    return wrapper


def _sanity_check(adatas: list[AnnData], slide_key: str = None):
    """
    Check that the AnnData objects are preprocessed correctly
    """
    assert hasattr(
        adatas, "__iter__"
    ), f"The input `adata` must be an AnnData object, or an iterable of AnnData objects. Found {type(adatas)}"
    assert all(isinstance(adata, AnnData) for adata in adatas), "All `adata` elements must be AnnData objects"

    count_raw = 0

    for adata in adatas:
        if Keys.SLIDE_ID in adata.obs:
            del adata.obs[Keys.SLIDE_ID]

        if slide_key is None:
            adata.obs[Keys.SLIDE_ID] = pd.Series(id(adata), index=adata.obs_names, dtype="category")
        else:
            assert slide_key in adata.obs, f"{slide_key=} must be in all adata.obs"
            values: pd.Series = f"{id(adata)}_" + adata.obs[slide_key].astype(str)
            adata.obs[Keys.SLIDE_ID] = values.astype("category")

        if adata.X.min() < 0:
            log.warn("Found some negative values in adata.X. We recommended having unscaled data (raw counts or log1p)")

        if adata.X.max() >= 10:
            count_raw += 1

            adata.layers[Keys.COUNTS_LAYER] = adata.X.copy()
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)

        if Keys.ADJ not in adata.obsp:
            spatial_neighbors(adata, radius=[0, Nums.DELAUNAY_RADIUS_TH], slide_key=slide_key)

        mean_distance = adata.obsp[Keys.ADJ].data.mean()

        warning_cs = "Your coordinate system may not be in microns, which would lead to unexpected behaviors. Read the documentation of `novae.utils.spatial_neighbors` to fix this."
        if mean_distance >= Nums.MEAN_DISTANCE_UPPER_TH_WARNING:
            log.warn(f"The mean distance between neighborhood cells is {mean_distance}, which is high. {warning_cs}")
        elif mean_distance <= Nums.MEAN_DISTANCE_LOWER_TH_WARNING:
            log.warn(f"The mean distance between neighborhood cells is {mean_distance}, which is low. {warning_cs}")
        else:
            mean_ngh = adata.obsp[Keys.ADJ].getnnz(axis=1).mean()

            if mean_ngh <= Nums.MEAN_NGH_TH_WARNING:
                log.warn(
                    f"The mean number of neighbors is {mean_ngh}, which is very low. Consider re-running `spatial_neighbors` with a different `radius` threshold."
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


def pretty_num_parameters(model: torch.nn.Module) -> str:
    n_params = sum(p.numel() for p in model.parameters())

    if n_params < 1_000_000:
        return f"{n_params / 1_000:.1f}K"

    return f"{n_params / 1_000_000:.1f}M"


def parse_device_args(accelerator: str = "cpu") -> torch.device:
    """Updated from scvi-tools"""
    connector = _AcceleratorConnector(accelerator=accelerator)
    _accelerator = connector._accelerator_flag
    _devices = connector._devices_flag

    if _accelerator == "cpu":
        return torch.device("cpu")

    if isinstance(_devices, list):
        device_idx = _devices[0]
    elif isinstance(_devices, str) and "," in _devices:
        device_idx = _devices.split(",")[0]
    else:
        device_idx = _devices

    return torch.device(f"{_accelerator}:{device_idx}")
