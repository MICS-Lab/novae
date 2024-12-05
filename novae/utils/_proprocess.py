import numpy as np
import pandas as pd
from anndata import AnnData

from .._constants import Nums
from . import iter_slides
from ._validate import _check_has_slide_id


def quantile_scaling(
    adata: AnnData | list[AnnData],
    multiplier: float = 5,
    quantile: float = 0.2,
    per_slide: bool = True,
) -> pd.DataFrame:
    """Preprocess fluorescence data from `adata.X` using quantiles of expression.
    For each column `X`, we compute `asinh(X / 5*Q(0.2, X))`, and store them back.

    Args:
        adata: An `AnnData` object, or a list of `AnnData` objects.
        multiplier: The multiplier for the quantile.
        quantile: The quantile to compute.
        per_slide: Whether to compute the quantile per slide. If `False`, the quantile is computed for each `AnnData` object.
    """
    _check_has_slide_id(adata)

    if isinstance(adata, list):
        for adata_ in adata:
            quantile_scaling(adata_, multiplier, quantile, per_slide=per_slide)
        return

    if not per_slide:
        return _quantile_scaling(adata, multiplier, quantile)

    for adata_ in iter_slides(adata):
        _quantile_scaling(adata_, multiplier, quantile)


def _quantile_scaling(adata: AnnData, multiplier: float, quantile: float):
    df = adata.to_df()

    divider = multiplier * np.quantile(df, quantile, axis=0)
    divider[divider == 0] = df.max(axis=0)[divider == 0] + Nums.EPS

    adata.X = np.arcsinh(df / divider)
