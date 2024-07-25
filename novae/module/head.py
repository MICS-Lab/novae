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

log = logging.getLogger(__name__)


class SwavHead(L.LightningModule):
    queue: None | Tensor  # (n_tissues, queue_size, num_prototypes)

    def __init__(
        self,
        output_size: int,
        num_prototypes: int,
        temperature: float,
        temperature_weight_proto: float,
    ):
        """SwavHead module, adapted from the paper "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments".

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
        self.temperature_weight_proto = temperature_weight_proto

        self.lambda_regularization = 0.0  # may be updated on_epoch_start

        self.prototypes = nn.Parameter(torch.empty((self.num_prototypes, self.output_size)))
        self.prototypes = nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5), mode="fan_out")
        self.normalize_prototypes()
        self.min_prototypes = int(num_prototypes * Nums.MIN_PROTOTYPES_RATIO)

        self.queue = None
        self.use_queue = False

        self._clustering = None
        self._clusters_levels = None

    def init_queue(self, tissue_names: list[str]) -> None:
        del self.queue

        shape = (len(tissue_names) + 1, Nums.QUEUE_SIZE, self.num_prototypes)
        self.register_buffer("queue", torch.full(shape, 1 / self.num_prototypes))

        self.tissue_label_encoder = {tissue: i for i, tissue in enumerate(tissue_names)}
        self.tissue_label_encoder[None] = len(tissue_names)

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

        out1 = F.normalize(out1, dim=1, p=2)  # (B, output_size)
        out2 = F.normalize(out2, dim=1, p=2)  # (B, output_size)

        scores1 = out1 @ self.prototypes.T  # (B, num_prototypes)
        scores2 = out2 @ self.prototypes.T  # (B, num_prototypes)

        ilocs = self.get_prototype_ilocs(scores1, tissue)
        scores1, scores2 = scores1[:, ilocs], scores2[:, ilocs]

        q1 = self.sinkhorn(scores1)  # (B, num_prototypes) or (B, len(ilocs))
        q2 = self.sinkhorn(scores2)  # (B, num_prototypes) or (B, len(ilocs))

        loss = -0.5 * (self.cross_entropy_loss(q1, scores2) + self.cross_entropy_loss(q2, scores1))

        return loss, _mean_entropy_normalized(q1)

    @torch.no_grad()
    def get_prototype_ilocs(self, scores: Tensor, tissue: str | None = None) -> Tensor:
        if self.queue is None:
            return ...

        tissue_index = self.tissue_label_encoder[tissue]
        tissue_weights = F.softmax(scores / self.temperature_weight_proto, dim=1).mean(0)

        self.queue[tissue_index, 1:] = self.queue[tissue_index, :-1].clone()
        self.queue[tissue_index, 0] = tissue_weights

        if not self.use_queue:
            return ...

        weights = self.sinkhorn(self.queue.mean(dim=1))[tissue_index]
        ilocs = torch.where(weights > 1 / self.num_prototypes)[0]

        return ilocs if len(ilocs) >= self.min_prototypes else torch.topk(weights, self.min_prototypes).indices

    @torch.no_grad()
    def sinkhorn(self, scores: Tensor) -> Tensor:
        """Apply the Sinkhorn-Knopp algorithm to the scores.

        Args:
            scores: The normalized embeddings projected into the prototypes

        Returns:
            The soft codes from the Sinkhorn-Knopp algorithm.
        """
        Q = torch.exp(scores / Nums.SWAV_EPSILON).t()  # (num_prototypes, B) for consistency with the paper
        Q /= torch.sum(Q)

        K, B = Q.shape

        for _ in range(Nums.SINKHORN_ITERATIONS):
            Q /= torch.sum(Q, dim=1, keepdim=True) + self.lambda_regularization
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True) + self.lambda_regularization
            Q /= B
        Q = (Q / Q.sum(dim=0, keepdim=True)).t()

        return Q

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
