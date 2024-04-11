from __future__ import annotations

import logging

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

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

    def forward(self, x: torch.Tensor, genes_indices: torch.Tensor) -> torch.Tensor:
        genes_embeddings = self.embedding(genes_indices)
        genes_embeddings = F.normalize(genes_embeddings, dim=0, p=2)

        return x @ genes_embeddings
