from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor, nn

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
    )  # (B x G x (C + E))


class InferenceHeadZIE(L.LightningModule):
    def __init__(self, cell_embedder: CellEmbedder, input_size: int, hidden_size: int = 32, n_layers: int = 5):
        super().__init__()
        self.cell_embedder = cell_embedder
        self.mlp = _mlp(input_size, hidden_size, n_layers, 2)

    def forward(self, z: Tensor, genes_embeddings: Tensor) -> Tensor:
        """
        z: (B x C)
        genes_embeddings: (G x E)
        """
        combined_embeddings = _combine_embeddings(z, genes_embeddings)  # (B x G x (C + E)

        logits = self.mlp(combined_embeddings)  # (B x G x 2)

        return logits[..., 0], logits[..., 1]  # pi_logits (B x G), lambda_logits (B x G)

    def loss(self, x: Tensor, z: Tensor, var_names: str | list[str]) -> Tensor:
        """Negative log-likelihood of the zero-inflated exponential distribution

        Args:
            x: Expressions of genes (B x G)
            z: Latent space (B x C)

        Returns:
            The mean negative log-likelihood
        """
        var_names = [var_names] if isinstance(var_names, str) else var_names
        genes_embeddings = self.cell_embedder.embedding(self.cell_embedder.genes_to_indices(var_names).to(self.device))

        pi_logits, lambda_logits = self(z, genes_embeddings)

        case_zero = -pi_logits
        case_non_zero = -lambda_logits + x * torch.exp(lambda_logits)
        return torch.where(x > 0, case_non_zero, case_zero).mean()


class InferenceHeadBaseline(L.LightningModule):
    def __init__(self, cell_embedder: CellEmbedder, input_size: int, hidden_size: int = 32, n_layers: int = 5):
        super().__init__()
        self.cell_embedder = cell_embedder
        self.mlp = _mlp(input_size, hidden_size, n_layers, 1)

    def forward(self, z: Tensor, genes_embeddings: Tensor) -> Tensor:
        """
        z: (B x C)
        genes_embeddings: (G x E)
        """
        combined_embeddings = _combine_embeddings(z, genes_embeddings)  # (B x G x (C + E)

        return self.mlp(combined_embeddings).squeeze(-1)  # predictions (B x G)

    def loss(self, x: Tensor, z: Tensor, var_names: str | list[str]) -> Tensor:
        var_names = [var_names] if isinstance(var_names, str) else var_names
        genes_embeddings = self.cell_embedder.embedding(self.cell_embedder.genes_to_indices(var_names).to(self.device))

        predictions = self(z, genes_embeddings)

        return F.mse_loss(x, predictions)
