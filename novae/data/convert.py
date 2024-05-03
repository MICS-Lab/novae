from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch import Tensor

from .._constants import EPS, IS_KNOWN_GENE_KEY, N_OBS_THRESHOLD


class AnnDataTorch:
    tensors: list[Tensor] | None
    var_names_list: list[pd.Index]

    def __init__(self, adatas: list[AnnData]):
        super().__init__()
        self.adatas = adatas

        self.tensors = None
        # Tensors are loaded in-memory for low numbers of cells
        if sum(adata.n_obs for adata in self.adatas) < N_OBS_THRESHOLD:
            self.tensors = [torch.tensor(self.array(adata)) for adata in self.adatas]

        self.var_names_list = [self.get_var_names(adata) for adata in self.adatas]

    def get_var_names(self, adata: AnnData) -> pd.Index:
        if IS_KNOWN_GENE_KEY in adata.var:
            return adata.var_names[adata.var[IS_KNOWN_GENE_KEY]]
        return adata.var_names

    def array(self, adata: AnnData) -> np.ndarray:
        if IS_KNOWN_GENE_KEY in adata.var:
            adata = adata[:, adata.var[IS_KNOWN_GENE_KEY]]

        X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        X = X.astype(np.float32)

        X = (X - adata.var["mean"].values) / (adata.var["std"].values + EPS)

        return X

    def __getitem__(self, item: tuple[int, slice]) -> tuple[Tensor, pd.Index]:
        adata_index, obs_indices = item

        if self.tensors is not None:
            return self.tensors[adata_index][obs_indices], self.var_names_list[adata_index]

        adata = self.adatas[adata_index]
        adata_view = adata[obs_indices]

        return torch.tensor(self.array(adata_view)), self.var_names_list[adata_index]
