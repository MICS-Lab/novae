from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from anndata import AnnData
from sklearn.decomposition import PCA
from torch import nn


class GenesEmbedding(pl.LightningModule):
    def __init__(self, gene_names: list[str], embedding_size: int) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.voc_size = len(gene_names)
        self.gene_to_index = {gene: i for i, gene in enumerate(gene_names)}

        self.embedding = nn.Embedding(self.voc_size, embedding_size)
        self.softmax = nn.Softmax(dim=0)

    def genes_to_indices(self, gene_names: list[str]) -> torch.Tensor:
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
