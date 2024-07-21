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
from . import CellEmbedder

log = logging.getLogger(__name__)


class SwavHead(L.LightningModule):
    QUEUE_SIZE = 4
    queue: None | Tensor

    def __init__(
        self,
        output_size: int,
        num_prototypes: int,
        temperature: float,
        lambda_regularization: float,
        epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
    ):
        """SwavHead module, adapted from in the paper "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments".

        Args:
            output_size: Size of the encoder embeddings.
            num_prototypes: Number of prototypes.
            temperature: Temperature used in the cross-entropy loss.
            epsilon: The entropy regularization term.
            sinkhorn_iterations: The number of Sinkhorn-Knopp iterations.
        """
        super().__init__()
        self.output_size = output_size
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.lambda_regularization = lambda_regularization
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations

        self.prototypes = nn.Parameter(torch.empty((self.num_prototypes, self.output_size)))
        self.prototypes = nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5), mode="fan_out")
        self.normalize_prototypes()

        self.queue = None
        self.use_queue = False

        self._clustering = None
        self._clusters_levels = None

    def init_queue(self, tissue_names: list[str]) -> None:
        del self.queue
        self.register_buffer("queue", torch.zeros(len(tissue_names), self.QUEUE_SIZE, self.num_prototypes))
        self.tissue_label_encoder = {tissue: i for i, tissue in enumerate(tissue_names)}

    @torch.no_grad()
    def normalize_prototypes(self):
        self.prototypes.data = F.normalize(self.prototypes.data, dim=1, p=2)

    def forward(self, out1: Tensor, out2: Tensor, tissue: str | None) -> Tensor:
        """Compute the SWAV loss for two batches of neighborhood graph views.

        Args:
            out1: Batch containing graphs embeddings `(B, output_size)`
            out2: Batch containing graphs embeddings `(B, output_size)`

        Returns:
            The SWAV loss
        """
        self.normalize_prototypes()

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        scores1 = out1 @ self.prototypes.T  # (B x num_prototypes)
        scores2 = out2 @ self.prototypes.T  # (B x num_prototypes)

        q1 = self.sinkhorn(scores1)  # (B x num_prototypes)
        q2 = self.sinkhorn(scores2)  # (B x num_prototypes)

        if tissue is not None:
            q1 *= self.get_tissue_weights(scores1, tissue)
            q1 /= q1.sum(dim=1, keepdim=True)
            q2 *= self.get_tissue_weights(scores2, tissue)
            q2 /= q2.sum(dim=1, keepdim=True)

        loss = -0.5 * (self.cross_entropy_loss(q1, scores2) + self.cross_entropy_loss(q2, scores1))

        return loss, _mean_entropy_normalized(q1)

    @torch.no_grad()
    def get_tissue_weights(self, scores: Tensor, tissue: str):
        tissue_index = self.tissue_label_encoder[tissue]
        tissue_weights = F.softmax(scores / (self.temperature / 4), dim=1).mean(0)
        self.queue[tissue_index, :-1] = self.queue[tissue_index, 1:].clone()
        self.queue[tissue_index, -1] = tissue_weights

        return self.sinkhorn(self.queue.mean(dim=1))[tissue_index] if self.use_queue else 1  # TODO: on init epoch?

    @torch.no_grad()
    def sinkhorn(self, scores: Tensor, tissue: str | None = None) -> Tensor:
        """Apply the Sinkhorn-Knopp algorithm to the scores.

        Args:
            scores: The normalized embeddings projected into the prototypes

        Returns:
            The soft codes from the Sinkhorn-Knopp algorithm.
        """
        if tissue:
            tissue_weights = self.queue[self.tissue_label_encoder[tissue]].mean(dim=0)

        Q = torch.exp(scores / self.epsilon).t()  # (num_prototypes x B) for consistency with notations from the paper
        Q /= torch.sum(Q)

        K, B = Q.shape

        for _ in range(self.sinkhorn_iterations):
            Q /= torch.sum(Q, dim=1, keepdim=True) + self.lambda_regularization
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True) + self.lambda_regularization
            Q /= B
        Q = (Q / Q.sum(dim=0, keepdim=True)).t()  # TODO: why changed B to sum?

        return Q if not tissue else Q * tissue_weights

    def cross_entropy_loss(self, q: Tensor, p: Tensor) -> Tensor:
        return torch.mean(torch.sum(q * F.log_softmax(p / self.temperature, dim=1), dim=1))

    def hierarchical_clustering(self) -> None:
        """
        Perform hierarchical clustering on the prototypes. Saves the full tree of clusters.
        """
        X = self.prototypes.data.numpy(force=True)  # (num_prototypes, output_size)

        self._clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            compute_full_tree=True,
            metric="cosine",
            linkage="average",
        )
        self._clustering.fit(X)

        self._clusters_levels = np.zeros((len(X), len(X)), dtype=np.uint16)
        self._clusters_levels[0] = np.arange(len(X))

        for i, (a, b) in enumerate(self._clustering.children_):
            clusters = self._clusters_levels[i]
            self._clusters_levels[i + 1] = clusters
            self._clusters_levels[i + 1, np.where((clusters == a) | (clusters == b))] = len(X) + i

    @property
    def clustering(self) -> AgglomerativeClustering:
        if self._clustering is None:
            self.hierarchical_clustering()
        return self._clustering

    def map_leaves_domains(self, series: pd.Series, n_classes: int) -> pd.Series:
        """Map leaves to the parent domain from the corresponding level of the hierarchical tree.

        Args:
            series: Leaves classes
            n_classes: Number of classes after mapping

        Returns:
            Series of classes (one among `n_classes`).
        """
        if self._clusters_levels is None:
            self.hierarchical_clustering()

        return series.map(lambda x: f"N{self._clusters_levels[-n_classes, int(x[1:])]}" if isinstance(x, str) else x)

    def reset_clustering(self) -> None:
        self._clustering = None
        self._clusters_levels = None

    def rotations_geodesic(self, centroids: np.ndarray, centroids_reference: np.ndarray) -> np.ndarray:
        """Computes the rotation matrices that transforms the centroids to the centroids_reference along the geodesic.

        Args:
            centroids: An array of size (..., output_size) of centroids of size `output_size`
            centroids_reference: An array of size (..., output_size) of centroids of size `output_size`

        Returns:
            An array of shape (..., output_size, output_size) of rotations matrices.
        """
        *left_shape, output_size = centroids.shape

        cos = (centroids * centroids_reference).sum(-1)
        sin = np.sin(np.arccos(cos))
        gamma = (cos - 1) / (sin + Nums.EPS) ** 2

        sum_centroids = centroids_reference + centroids

        identities = np.zeros((*left_shape, output_size, output_size))
        identities[..., np.arange(output_size), np.arange(output_size)] = 1

        return (
            identities
            + gamma[..., None, None] * sum_centroids[..., None, :] * sum_centroids[..., None]
            + 2 * centroids_reference[..., None] * centroids[..., None, :]
        )


@torch.no_grad()
def _mean_entropy_normalized(q: Tensor) -> Tensor:
    entropy = -(q * torch.log2(q + Nums.EPS)).sum(-1)
    max_entropy = torch.log2(torch.tensor(q.shape[-1]))
    return (entropy / max_entropy).mean()


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
