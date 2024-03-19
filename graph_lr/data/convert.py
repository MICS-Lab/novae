from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData
from torch import Tensor

from .._constants import EPS, N_OBS_THRESHOLD


class AnnDataTorch:
    def __init__(self, adatas: list[AnnData]):
        self.adatas = adatas

        self.tensors = None
        # Tensors are loaded in-memory for low numbers of cells
        if sum(adata.n_obs for adata in self.adatas) < N_OBS_THRESHOLD:
            self.tensors = [self.tensor(adata) for adata in self.adatas]

    def array(self, adata: AnnData) -> np.ndarray:
        X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        return X.astype(np.float32)

    def tensor(self, adata: AnnData) -> Tensor:
        X = self.array(adata)
        X = (X - adata.var["mean"].values) / (adata.var["std"].values + EPS)
        return torch.tensor(X)

    def __getitem__(self, item: tuple[int, slice]) -> torch.Tensor:
        adata_index, obs_indices = item

        if self.tensors is not None:
            return self.tensors[adata_index][obs_indices]

        adata = self.adatas[adata_index]
        adata_view = adata[obs_indices]

        return self.tensor(adata_view)
