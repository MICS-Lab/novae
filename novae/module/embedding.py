from __future__ import annotations

import logging

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from torch import nn
from torch_geometric.data import Data

from ..utils import lower_var_names

log = logging.getLogger(__name__)


class GenesEmbedding(L.LightningModule):
    def __init__(self, gene_names: list[str], embedding_size: int) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.vocabulary = gene_names
        self.voc_size = len(self.vocabulary)
        self.gene_to_index = {gene: i for i, gene in enumerate(self.vocabulary)}

        self.embedding = nn.Embedding(self.voc_size, embedding_size)

    def genes_to_indices(self, gene_names: pd.Index, as_torch: bool = True) -> torch.Tensor:
        indices = [self.gene_to_index[gene] for gene in lower_var_names(gene_names)]

        if as_torch:
            return torch.tensor(indices, dtype=torch.long, device=self.device)
        return np.array(indices, dtype=np.int16)

    def forward(self, data: Data) -> Data:
        genes_embeddings = self.embedding(data.genes_indices[0])
        genes_embeddings = F.normalize(genes_embeddings, dim=0, p=2)

        data.x = data.x @ genes_embeddings
        return data

    def pca_init(self, adatas: list[AnnData] | None):
        if adatas is None:
            return

        log.info("Running PCA embedding initialization")

        if len(adatas) > 1:
            # TODO: make it for any number of adatas, with different panel sizes
            log.warn("Shared PCA not implemented yet")

        adata = max(adatas, key=lambda adata: adata.n_vars)
        X = adata.X.toarray() if issparse(adata.X) else adata.X

        if X.shape[1] <= self.embedding_size:
            log.warn(f"PCA with {self.embedding_size} components can not be run on shape {X.shape}")
            return

        pca = PCA(n_components=self.embedding_size)
        pca.fit(X.astype(np.float32))

        indices = self.genes_to_indices(adata.var_names)
        self.embedding.weight.data[indices] = torch.tensor(pca.components_.T)

        known_var_names = lower_var_names(adata.var_names)

        for other_adata in adatas:
            other_var_names = lower_var_names(other_adata.var_names)
            where_in = np.isin(other_var_names, known_var_names)

            if where_in.all():
                continue

            X = other_adata[:, where_in].X.toarray().T
            Y = other_adata[:, ~where_in].X.toarray().T

            tree = KDTree(X)
            _, ind = tree.query(Y, k=1)
            neighbor_indices = self.genes_to_indices(other_adata[:, where_in].var_names[ind])

            indices = self.genes_to_indices(other_adata[:, ~where_in].var_names)
            self.embedding.weight.data[indices] = self.embedding.weight.data[neighbor_indices].clone()
