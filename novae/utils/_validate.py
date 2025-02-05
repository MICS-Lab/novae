import logging

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from .. import settings
from .._constants import Keys, Nums
from . import lower_var_names

log = logging.getLogger(__name__)


def prepare_adatas(
    adata: AnnData | list[AnnData] | None,
    var_names: set | list[str] | None = None,
) -> list[AnnData]:
    """Ensure the AnnData objects are ready to be used by the model.

    Note:
        It performs the following operations:

        - Preprocess the data if needed (e.g. normalize, log1p), in which case raw counts are saved in `adata.layers['counts']`
        - Compute the mean and std of each gene
        - Save which genes are highly variable, in case the number of genes is too high
        - If using a pretrained model, save which genes are known by the model

    Args:
        adata: An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.
        var_names: Only used when loading a pretrained model. Do not use it yourself.

    Returns:
        A list of `AnnData` objects ready to be used by the model. If only one `adata` object is provided, it will be wrapped in a list.
    """
    assert adata is not None or var_names is not None, "One of `adata` and `var_names` must not be None"

    if adata is None:
        return None, var_names

    if isinstance(adata, AnnData):
        adatas = [adata]
    elif isinstance(adata, list):
        adatas = adata
    else:
        raise ValueError(f"Invalid type for `adata`: {type(adata)}")

    assert len(adatas) > 0, "No `adata` object found. Please provide an AnnData object, or a list of AnnData objects."

    assert all(
        Keys.ADJ in adata.obsp for adata in adatas
    ), "You need to first run `novae.spatial_neighbors` to compute cell neighbors."

    _check_has_slide_id(adatas)
    _standardize_adatas(adatas)  # log1p + spatial_neighbors
    if settings.auto_preprocessing:
        _lookup_highly_variable_genes(adatas)
    _select_novae_genes(adatas, var_names)

    if var_names is None:
        var_names = _genes_union(adatas, among_used=True)

    return adatas, var_names


def _check_has_slide_id(adata: AnnData | list[AnnData]):
    if not isinstance(adata, AnnData):
        for adata_ in adata:
            _check_has_slide_id(adata_)
        return
    assert (
        Keys.SLIDE_ID in adata.obs
    ), f"Column `adata.obs['{Keys.SLIDE_ID}']` not found. Run `novae.spatial_neighbors` first."


def _standardize_adatas(adatas: list[AnnData]):
    """
    Make sure all AnnData objects are preprocessed correctly and have a Delaunay graph
    """
    assert hasattr(
        adatas, "__iter__"
    ), f"The input `adata` must be an AnnData object, or an iterable of AnnData objects. Found {type(adatas)}"
    assert all(isinstance(adata, AnnData) for adata in adatas), "All `adata` elements must be AnnData objects"

    count_raw = 0

    for adata in adatas:
        if adata.X.min() < 0:
            log.warning(
                "Found some negative values in adata.X. We recommended having unscaled data (raw counts or log1p)"
            )

        max_value = adata.X.max()
        if settings.auto_preprocessing and max_value >= 10:
            if int(max_value) == max_value:  # counts
                count_raw += 1

                adata.layers[Keys.COUNTS_LAYER] = adata.X.copy()
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
            else:
                log.warning(
                    "adata.X has high values. We recommend processing the data before running the model (e.g., with sc.pp.log1p)."
                )
    if count_raw:
        log.info(
            f"Preprocessed {count_raw} adata object(s) with sc.pp.normalize_total and sc.pp.log1p (raw counts were saved in adata.layers['{Keys.COUNTS_LAYER}'])"
        )


def check_has_spatial_adjancency(adata: AnnData):
    assert "spatial" in adata.obsm, "No spatial coordinates found in `adata.obsm['spatial']`"
    assert Keys.ADJ in adata.obsp, (
        "No spatial adjacency found in `adata.obsp['spatial_distances']`." "Consider running `novae.spatial_neighbors`"
    )


def _lookup_highly_variable_genes(adatas: list[AnnData]):
    if len(adatas) == 1 or _is_multi_panel(adatas):
        for adata in adatas:
            _highly_variable_genes(adata)
        return

    adata_ = anndata.concat(adatas)

    assert adata_.n_vars == adatas[0].n_vars, "Same panel used but the gene names don't have the same case"

    _highly_variable_genes(adata_, set_default_true=True)

    highly_variable_genes = adata_.var_names[adata_.var[Keys.HIGHLY_VARIABLE]]

    for adata in adatas:
        adata.var[Keys.HIGHLY_VARIABLE] = False
        adata.var.loc[highly_variable_genes, Keys.HIGHLY_VARIABLE] = True


def _highly_variable_genes(adata: AnnData, set_default_true: bool = False):
    if Keys.HIGHLY_VARIABLE in adata.var:  # already computed
        return

    if adata.n_vars <= Nums.MIN_GENES_FOR_HVG:  # if too few genes, keep all genes
        if set_default_true:
            adata.var[Keys.HIGHLY_VARIABLE] = True
        return

    sc.pp.highly_variable_genes(adata)

    n_hvg = adata.var[Keys.HIGHLY_VARIABLE].sum()

    if n_hvg < Nums.MIN_GENES:
        log.warning(f"Only {n_hvg} highly variable genes were found.")

    if n_hvg < Nums.N_HVG_THRESHOLD and n_hvg < adata.n_vars // 2:
        sc.pp.highly_variable_genes(adata, n_top_genes=adata.n_vars // 2)  # keep at least half of the genes


def _select_novae_genes(adatas: list[AnnData], var_names: set | list[str] | None) -> None:
    for adata in adatas:
        _lookup_known_genes(adata, var_names)

        adata.var[Keys.USE_GENE] = _var_or_true(adata, Keys.HIGHLY_VARIABLE) & _var_or_true(adata, Keys.IS_KNOWN_GENE)

        n_used = adata.var[Keys.USE_GENE].sum()
        assert (
            n_used >= Nums.MIN_GENES
        ), f"Too few genes ({n_used}) are both (i) known by the model and (ii) highly variable."


def _lookup_known_genes(adata: AnnData, var_names: set | list[str] | None):
    if var_names is None or Keys.IS_KNOWN_GENE in adata.var:
        return

    adata.var[Keys.IS_KNOWN_GENE] = np.isin(lower_var_names(adata.var_names), list(var_names))

    n_known = sum(adata.var[Keys.IS_KNOWN_GENE])
    assert n_known >= Nums.MIN_GENES, f"Too few genes ({n_known}) are known by the model."
    if n_known / adata.n_vars < 0.50:
        log.warning(f"Only {n_known / adata.n_vars:.1%} of genes are known by the model.")


def _genes_union(adatas: list[AnnData], among_used: bool = False) -> list[str]:
    if among_used:
        var_names_list = [adata.var_names[adata.var[Keys.USE_GENE]] for adata in adatas]
    else:
        var_names_list = [adata.var_names for adata in adatas]

    return list(set.union(*[set(lower_var_names(var_names)) for var_names in var_names_list]))


def _var_or_true(adata: AnnData, key: str) -> pd.Series | bool:
    return adata.var[key] if key in adata.var else True


def _is_multi_panel(adatas: list[AnnData]) -> bool:
    if len(adatas) == 1:
        return False

    first_panel = sorted(lower_var_names(adatas[0].var_names))

    for adata in adatas[1:]:
        if sorted(lower_var_names(adata.var_names)) != first_panel:
            return True

    return False


ERROR_ADVICE_OBS_KEY = "Please run `model.assign_domains(...)` first"


def _available_domains_key(adata: AnnData) -> set[int]:
    return set(adata.obs.columns[adata.obs.columns.str.startswith(Keys.DOMAINS_PREFIX)])


def _shared_domains_keys(adatas: list[AnnData]) -> set[int]:
    available_keys = [_available_domains_key(adata) for adata in adatas]
    assert any(available_keys), f"No Novae domains available. {ERROR_ADVICE_OBS_KEY}"

    available_keys = set.intersection(*available_keys)
    assert available_keys, f"No common Novae domains available. {ERROR_ADVICE_OBS_KEY}"

    return available_keys


def check_available_domains_key(adatas: list[AnnData], obs_key: str | None) -> str:
    available_obs_keys = _shared_domains_keys(adatas)

    assert len(available_obs_keys), f"No Novae domains available. {ERROR_ADVICE_OBS_KEY}"

    if obs_key is not None:
        assert all(
            obs_key in adata.obs for adata in adatas
        ), f"Novae domains '{obs_key}' not available in all AnnData objects. {ERROR_ADVICE_OBS_KEY}. Or consider using one of {available_obs_keys} instead."
        return obs_key

    obs_key = list(available_obs_keys)[-1]
    log.info(f"Using {obs_key=} by default.")
    return obs_key


def check_slide_name_key(adatas: AnnData | list[AnnData], slide_name_key: str | None) -> str:
    if isinstance(adatas, AnnData):
        return check_slide_name_key([adatas], slide_name_key)

    _check_has_slide_id(adatas)

    if slide_name_key is None:
        return Keys.SLIDE_ID

    for adata in adatas:
        assert slide_name_key in adata.obs, f"Column '{slide_name_key}' not found in adata.obs"
        counts = adata.obs.groupby(Keys.SLIDE_ID, observed=True)[slide_name_key].value_counts().unstack()
        assert ((counts > 0).sum(1) == 1).all(), f"Column '{slide_name_key}' is not unique per slide"

    return slide_name_key
