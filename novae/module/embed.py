import json
import logging
from pathlib import Path

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

from .. import utils
from .._constants import Keys

log = logging.getLogger(__name__)


class CellEmbedder(L.LightningModule):
    """Convert a cell into an embedding using a gene embedding matrix."""

    def __init__(
        self,
        gene_names: list[str] | dict[str, int],
        embedding_size: int | None,
        embedding: torch.Tensor | None = None,
    ) -> None:
        """

        Args:
            gene_names: Name of the genes to be used in the embedding, or dictionnary of index to name.
            embedding_size: Size of the embeddings of the genes (`E` in the article). Optional if `embedding` is provided.
            embedding: Optional pre-trained embedding matrix. If provided, `embedding_size` shouldn't be provided.
        """
        super().__init__()
        assert (embedding_size is None) ^ (embedding is None), "Either embedding_size or embedding must be provided"

        if isinstance(gene_names, dict):
            self.gene_to_index = {gene.lower(): index for gene, index in gene_names.items()}
            self.gene_names = sorted(self.gene_to_index, key=self.gene_to_index.get)
            _check_gene_to_index(self.gene_to_index)
        else:
            self.gene_names = [gene.lower() for gene in gene_names]
            self.gene_to_index = {gene: i for i, gene in enumerate(self.gene_names)}

        self.voc_size = len(self.gene_names)

        if embedding is None:
            self.embedding_size = embedding_size
            self.embedding = nn.Embedding(self.voc_size, embedding_size)
        else:
            self.embedding_size = embedding.size(1)
            self.embedding = nn.Embedding.from_pretrained(embedding)

        self.linear = nn.Linear(self.embedding_size, self.embedding_size)
        self._init_linear()

    @torch.no_grad()
    def _init_linear(self):
        self.linear.weight.data.copy_(torch.eye(self.embedding_size))
        self.linear.bias.data.zero_()

    @classmethod
    def from_scgpt_embedding(cls, scgpt_model_dir: str) -> "CellEmbedder":
        """Initialize the CellEmbedder from a scGPT pretrained model directory.

        Args:
            scgpt_model_dir: Path to a directory containing a scGPT checkpoint, i.e. a `vocab.json` and a `best_model.pt` file.

        Returns:
            A CellEmbedder instance.
        """
        scgpt_model_dir = Path(scgpt_model_dir)

        vocab_file = scgpt_model_dir / "vocab.json"

        with open(vocab_file, "r") as file:
            gene_to_index: dict[str, int] = json.load(file)

        checkpoint = torch.load(scgpt_model_dir / "best_model.pt", map_location=torch.device("cpu"))
        embedding = checkpoint["encoder.embedding.weight"]

        return cls(gene_to_index, None, embedding=embedding)

    def genes_to_indices(self, gene_names: pd.Index | list[str], as_torch: bool = True) -> torch.Tensor | np.ndarray:
        """Convert gene names to their corresponding indices.

        Args:
            gene_names: Names of the gene names to convert.
            as_torch: Whether to return a `torch` tensor or a `numpy` array.

        Returns:
            A tensor or array of gene indices.
        """
        indices = [self.gene_to_index[gene] for gene in utils.lower_var_names(gene_names)]

        if as_torch:
            return torch.tensor(indices, dtype=torch.long)

        return np.array(indices, dtype=np.int16)

    def forward(self, data: Data) -> Data:
        """Embed the input data.

        Args:
            data: A Pytorch Geometric `Data` object representing a batch of `B` graphs. The number of node features is variable.

        Returns:
            data: A Pytorch Geometric `Data` object representing a batch of `B` graphs. Each node now has a size of `E`.
        """
        genes_embeddings = self.embedding(data.genes_indices[0])
        genes_embeddings = self.linear(genes_embeddings)
        genes_embeddings = F.normalize(genes_embeddings, dim=0, p=2)

        data.x = data.x @ genes_embeddings
        return data

    def pca_init(self, adatas: list[AnnData] | None):
        """Initialize the Noave embeddings with PCA components.

        Args:
            adatas: A list of `AnnData` objects to use for PCA initialization.
        """
        if adatas is None:
            return

        adatas = [adata[:, adata.var[Keys.USE_GENE]] for adata in adatas]

        adata = max(adatas, key=lambda adata: adata.n_vars)

        if adata.X.shape[1] <= self.embedding_size:
            log.warning(
                f"PCA with {self.embedding_size} components can not be run on shape {adata.X.shape}.\nTo use PCA initialization, set a lower `embedding_size` (<{adata.X.shape[1]}) in novae.Novae()."
            )
            return

        X = adata.X.toarray() if issparse(adata.X) else adata.X

        log.info("Running PCA embedding initialization")

        pca = PCA(n_components=self.embedding_size)
        pca.fit(X.astype(np.float32))

        indices = self.genes_to_indices(adata.var_names)
        self.embedding.weight.data[indices] = torch.tensor(pca.components_.T)

        known_var_names = utils.lower_var_names(adata.var_names)

        for other_adata in adatas:
            other_var_names = utils.lower_var_names(other_adata.var_names)
            where_in = np.isin(other_var_names, known_var_names)

            if where_in.all():
                continue

            X = other_adata[:, where_in].X.toarray().T
            Y = other_adata[:, ~where_in].X.toarray().T

            tree = KDTree(X)
            _, ind = tree.query(Y, k=1)
            neighbor_indices = self.genes_to_indices(other_adata[:, where_in].var_names[ind[:, 0]])

            indices = self.genes_to_indices(other_adata[:, ~where_in].var_names)
            self.embedding.weight.data[indices] = self.embedding.weight.data[neighbor_indices].clone()


def _check_gene_to_index(gene_to_index: dict[str, int]):
    values = list(set(gene_to_index.values()))

    assert len(values) == len(gene_to_index), "gene_to_index should be a dictionnary with unique values"

    assert (
        min(values) == 0 and max(values) == len(values) - 1
    ), "gene_to_index should be a dictionnary with continuous indices starting from 0"
