from __future__ import annotations

import logging
import math

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from torch import Tensor, nn

from .._constants import Nums
from .embedding import GenesEmbedding

log = logging.getLogger(__name__)


class SwavHead(L.LightningModule):
    def __init__(self, out_channels: int, num_prototypes: int, temperature: float):
        super().__init__()
        self.out_channels = out_channels
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        self.prototypes = nn.Parameter(torch.empty((self.num_prototypes, self.out_channels)))
        self.prototypes = nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5), mode="fan_out")
        self.normalize_prototypes()

        self.clusters_levels = None

    @torch.no_grad()
    def normalize_prototypes(self):
        self.prototypes.data = F.normalize(self.prototypes.data, dim=1, p=2)

    def forward(self, out1: Tensor, out2: Tensor) -> Tensor:
        """
        out1, out2: (B x out_channels)
        """
        self.normalize_prototypes()

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        scores1 = out1 @ self.prototypes.T  # (B x num_prototypes)
        scores2 = out2 @ self.prototypes.T  # (B x num_prototypes)

        q1 = self.sinkhorn(scores1)
        q2 = self.sinkhorn(scores2)

        return -0.5 * (self.cross_entropy_loss(q1, scores2) + self.cross_entropy_loss(q2, scores1))

    @torch.no_grad()
    def sinkhorn(self, out: Tensor, epsilon: float = 0.05, sinkhorn_iterations: int = 3) -> Tensor:
        """
        out: (B x num_prototypes)
        """
        Q = torch.exp(out / epsilon).t()  # (num_prototypes x B) for consistency with notations from the paper
        Q /= torch.sum(Q)

        K, B = Q.shape

        for _ in range(sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()

    def cross_entropy_loss(self, q: Tensor, p: Tensor) -> Tensor:
        return torch.mean(torch.sum(q * F.log_softmax(p / self.temperature, dim=1), dim=1))

    def hierarchical_clustering(self) -> None:
        X = self.prototypes.data.numpy(force=True)  # (num_prototypes, out_channels)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            compute_full_tree=True,
            metric="cosine",
            linkage="average",
        )
        clustering.fit(X)

        self.clusters_levels = np.zeros((len(X), len(X)), dtype=np.uint16)
        self.clusters_levels[0] = np.arange(len(X))

        for i, (a, b) in enumerate(clustering.children_):
            clusters = self.clusters_levels[i]
            self.clusters_levels[i + 1] = clusters
            self.clusters_levels[i + 1, np.where((clusters == a) | (clusters == b))] = len(X) + i

    def assign_classes_level(self, series: pd.Series, n_classes: int) -> pd.Series:
        if self.clusters_levels is None:
            self.hierarchical_clustering()

        return series.map(lambda x: x if np.isnan(float(x)) else str(self.clusters_levels[-n_classes, int(x)]))

    def rotations_geodesic(self, centroids: np.ndarray, centroids_reference: np.ndarray) -> np.ndarray:
        """Computes the rotation matrices that transforms the centroids to the centroids_reference along the geodesic.

        Args:
            centroids: An array of size (..., out_channels) of centroids of size `out_channels`
            centroids_reference: An array of size (..., out_channels) of centroids of size `out_channels`

        Returns:
            An array of shape (..., out_channels, out_channels) of rotations matrices.
        """
        *left_shape, out_channels = centroids.shape

        cos = (centroids * centroids_reference).sum(-1)
        sin = np.sin(np.arccos(cos))
        gamma = (cos - 1) / (sin + Nums.EPS) ** 2

        sum_centroids = centroids_reference + centroids

        identities = np.zeros((*left_shape, out_channels, out_channels))
        identities[..., np.arange(out_channels), np.arange(out_channels)] = 1

        return (
            identities
            + gamma[..., None, None] * sum_centroids[..., None, :] * sum_centroids[..., None]
            + 2 * centroids_reference[..., None] * centroids[..., None, :]
        )


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
    def __init__(self, genes_embedding: GenesEmbedding, input_size: int, hidden_size: int = 32, n_layers: int = 5):
        super().__init__()
        self.genes_embedding = genes_embedding
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
        genes_embeddings = self.genes_embedding.embedding(self.genes_embedding.genes_to_indices(var_names))

        pi_logits, lambda_logits = self(z, genes_embeddings)

        case_zero = -pi_logits
        case_non_zero = -lambda_logits + x * torch.exp(lambda_logits)
        return torch.where(x > 0, case_non_zero, case_zero).mean()


class InferenceHeadBaseline(L.LightningModule):
    def __init__(self, genes_embedding: GenesEmbedding, input_size: int, hidden_size: int = 32, n_layers: int = 5):
        super().__init__()
        self.genes_embedding = genes_embedding
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
        genes_embeddings = self.genes_embedding.embedding(self.genes_embedding.genes_to_indices(var_names))

        predictions = self(z, genes_embeddings)

        return F.mse_loss(x, predictions)
