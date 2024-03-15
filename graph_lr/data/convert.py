from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData

from .._constants import EPS


class AnnDataTorch:
    def __init__(self, adatas: list[AnnData]):
        self.adatas = adatas

    def __getitem__(self, item: tuple[int, slice]) -> torch.Tensor:
        adata_index, obs_indices = item

        adata = self.adatas[adata_index]
        adata_view = adata[obs_indices]

        X: np.ndarray = adata_view.X if isinstance(adata.X, np.ndarray) else adata_view.X.toarray()
        X = (X - adata.var["mean"].values) / (adata.var["std"].values + EPS)

        return torch.tensor(X.astype(np.float32))
