from __future__ import annotations

import lightning as L
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from sklearn.decomposition import PCA
from torch import nn

from ..utils import lower_var_names


class GenesEmbedding(L.LightningModule):
    def __init__(self, gene_names: list[str], embedding_size: int) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.vocabulary = gene_names
        self.voc_size = len(self.vocabulary)
        self.gene_to_index = {gene: i for i, gene in enumerate(self.vocabulary)}

        self.embedding = nn.Embedding(self.voc_size, embedding_size)
        self.softmax = nn.Softmax(dim=0)

    def genes_to_indices(self, gene_names: pd.Index) -> torch.Tensor:
        gene_names = lower_var_names(gene_names)
        return torch.tensor([self.gene_to_index[gene] for gene in gene_names], dtype=torch.long)

    def forward(self, x: torch.Tensor, genes_indices: torch.Tensor) -> torch.Tensor:
        genes_embeddings = self.embedding(genes_indices)
        genes_embeddings = F.normalize(genes_embeddings, dim=0, p=2)

        return x @ genes_embeddings

    def pca_init(self, adatas: list[AnnData]):
        # PCA init embeddings (valid only if x centered)
        # TODO: handle adatas
        pca = PCA(n_components=self.embedding_size)
        pca.fit(self.x_numpy)
        self.embedding.weight.data = torch.tensor(pca.components_.T)
