from __future__ import annotations

import lightning as L
import torch
from torch import Tensor, nn, optim
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MaxAggregation

from .. import utils
from . import CellEmbedder


def _mlp(input_size: int, hidden_size: int, n_layers: int, output_size: int) -> nn.Sequential:
    layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
    for _ in range(n_layers):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
    layers.append(nn.Linear(hidden_size, output_size))

    return nn.Sequential(*layers)


def _combine_embeddings(z: Tensor, genes_embeddings: Tensor) -> Tensor:
    n_obs, n_genes = len(z), len(genes_embeddings)

    return torch.cat(
        [z.unsqueeze(1).expand(-1, n_genes, -1), genes_embeddings.unsqueeze(0).expand(n_obs, -1, -1)], dim=-1
    )  # (B, G, (C + E))


class InferenceHeadPoisson(L.LightningModule):
    def __init__(self, cell_embedder: CellEmbedder, input_size: int, hidden_size: int = 64, n_layers: int = 5):
        super().__init__()
        self.cell_embedder = cell_embedder
        self.mlp = _mlp(input_size, hidden_size, n_layers, 1)

        self.poisson_nllloss = nn.PoissonNLLLoss(log_input=True)

    @utils.format_docs
    def forward(self, z: Tensor, genes_embeddings: Tensor) -> Tensor:
        """Compute the poisson parameter based on the cells representations and the genes embeddings.

        Args:
            {z}
            genes_embeddings: Embedding of `G` genes. Size `(G, E)`.

        Returns:
            The mean negative log-likelihood
        """
        combined_embeddings = _combine_embeddings(z, genes_embeddings)  # (B, G, (O + E)

        return self.mlp(combined_embeddings).squeeze(-1)  # (B, G)

    @utils.format_docs
    def loss(self, x: Tensor, z: Tensor, genes_indices: Tensor) -> Tensor:
        """Negative log-likelihood of the zero-inflated exponential distribution

        Args:
            x: Expressions of genes `(B, G)` as counts.
            {z}
            genes_indices: Tensor of gene indices to be predicted.

        Returns:
            The mean Poisson negative log-likelihood.
        """
        with torch.no_grad():
            genes_embeddings = self.cell_embedder.embedding(genes_indices)

        logits = self(z, genes_embeddings)

        return self.poisson_nllloss(logits, x)

    def infer(self, z: Tensor, genes_embeddings: Tensor) -> Tensor:
        logits = self(z, genes_embeddings)
        return torch.exp(logits)


class InferenceModel(L.LightningModule):
    def __init__(
        self,
        novae_model: L.LightningModule,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.novae_model = novae_model
        self.head = InferenceHeadPoisson(
            novae_model.cell_embedder, novae_model.hparams.output_size + novae_model.hparams.embedding_size
        )
        self.lr = lr

        self.max_aggregation = MaxAggregation()

    def forward(self, batch: dict[str, Data]) -> dict[str, Tensor]:
        with torch.no_grad():
            data = batch["main"]
            z = self.novae_model(batch)["main"]  # (B, O)
            x = self.max_aggregation(data.counts, index=data.batch)  # (B, G)
            genes_indices = data.counts_genes_indices  # (B, G)

        return sum([self.head.loss(x[[i]], z[[i]], genes_indices[i]) for i in range(len(x))])

    def training_step(self, batch: dict[str, Data], batch_idx: int) -> Tensor:
        loss = self.forward(batch)
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=True,
            batch_size=self.novae_model.hparams.batch_size,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
