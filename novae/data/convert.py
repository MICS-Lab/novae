from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData
from torch import Tensor

from .._constants import Keys, Nums
from ..module import CellEmbedder


class AnnDataTorch:
    tensors: list[Tensor] | None
    genes_indices_list: list[Tensor]

    def __init__(self, adatas: list[AnnData], cell_embedder: CellEmbedder):
        super().__init__()
        self.adatas = adatas
        self.cell_embedder = cell_embedder

        self.tensors = None
        # Tensors are loaded in-memory for low numbers of cells
        if sum(adata.n_obs for adata in self.adatas) < Nums.N_OBS_THRESHOLD:
            self.tensors = [torch.tensor(self.array(adata)) for adata in self.adatas]

        self.genes_indices_list = [self._adata_to_genes_indices(adata) for adata in self.adatas]

    def _adata_to_genes_indices(self, adata: AnnData) -> Tensor:
        return self.cell_embedder.genes_to_indices(adata.var_names[adata.var[Keys.USE_GENE]])[None, :]

    def array(self, adata: AnnData) -> np.ndarray:
        adata = adata[:, adata.var[Keys.USE_GENE]]

        X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        X = X.astype(np.float32)

        X = (X - adata.var["mean"].values) / (adata.var["std"].values + Nums.EPS)

        return X

    def __getitem__(self, item: tuple[int, slice]) -> tuple[Tensor, Tensor]:
        adata_index, obs_indices = item

        if self.tensors is not None:
            return self.tensors[adata_index][obs_indices], self.genes_indices_list[adata_index]

        adata = self.adatas[adata_index]
        adata_view = adata[obs_indices]

        return torch.tensor(self.array(adata_view)), self.genes_indices_list[adata_index]
