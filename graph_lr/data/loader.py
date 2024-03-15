from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData

from .._constants import EPS


class AnnDataLoader:
    def __init__(self, adatas: list[AnnData]):
        self.adatas = adatas

    def __getitem__(self, arg: tuple[int, slice]) -> torch.Tensor:
        i, indices = arg

        adata = self.adatas[i]
        adata_view = adata[indices]

        X: np.ndarray = adata_view.X if isinstance(adata.X, np.ndarray) else adata_view.X.toarray()
        X = (X - adata.var["mean"].values) / (adata.var["std"].values + EPS)

        return torch.tensor(X.astype(np.float32))
