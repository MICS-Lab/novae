from __future__ import annotations

import logging
import math

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from torch import nn

from .._constants import EPS

log = logging.getLogger(__name__)


class SwavHead(L.LightningModule):
    def __init__(
        self,
        out_channels: int,
        num_prototypes: int,
        temperature: float = 0.1,
        queue_size: int | None = None,
        epoch_queue_starts: int = 20,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.queue_size = queue_size
        self.epoch_queue_starts = epoch_queue_starts

        if self.queue_size is not None:
            self.register_buffer("queue", torch.zeros((self.queue_size, out_channels), dtype=torch.float32))
        else:
            self.queue = None

        self.prototypes = nn.Parameter(torch.empty((self.out_channels, self.num_prototypes)))
        self.prototypes = nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))
        self.normalize_prototypes()

        self.clusters_levels = None

    @property
    def use_queue(self) -> bool:
        return self.queue is not None and self.current_epoch >= self.epoch_queue_starts

    def init_prototypes_sample(self, X: torch.Tensor):
        log.info(f"Running sample init on shape {X.shape} for {self.num_prototypes} proto")
        self.prototypes.data = X[torch.randperm(X.size()[0])[: self.num_prototypes]].detach().clone().T
        log.info("done")

    def normalize_prototypes(self):
        self.prototypes.data = F.normalize(self.prototypes.data, dim=0, p=2)

    def forward(self, out1, out2):
        self.normalize_prototypes()

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        scores1 = out1 @ self.prototypes
        scores2 = out2 @ self.prototypes

        if self.use_queue:
            scores_queued1 = torch.cat([scores1.detach(), self.queue @ self.prototypes])
            scores_queued2 = torch.cat([scores2.detach(), self.queue @ self.prototypes])

            q1 = self.sinkhorn(scores_queued1)[: len(scores1)]
            q2 = self.sinkhorn(scores_queued2)[: len(scores2)]
        else:
            q1 = self.sinkhorn(scores1)
            q2 = self.sinkhorn(scores2)

        if self.queue is not None:
            n = len(out1)
            self.queue[n:] = self.queue[:-n].clone()
            self.queue[:n] = out1.detach()

        return -0.5 * (self.cross_entropy_loss(q1, scores2) + self.cross_entropy_loss(q2, scores1))

    @torch.no_grad()
    def sinkhorn(self, out, epsilon: float = 0.05, sinkhorn_iterations: int = 3):
        """Q is K-by-B for consistency with notations from the paper (out: B*K)"""
        Q = out  # - out.max() # remove comment to make it numerically more stable
        Q = torch.exp(Q / epsilon).t()
        Q /= torch.sum(Q)

        B = Q.shape[1]
        K = Q.shape[0]

        for _ in range(sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()

    def cross_entropy_loss(self, q, p):
        return torch.mean(torch.sum(q * F.log_softmax(p / self.temperature, dim=1), dim=1))

    def hierarchical_clustering(self):
        X = self.prototypes.data.T.numpy(force=True)  # shape (n_proto, out_channels)

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
        gamma = (cos - 1) / (sin + EPS) ** 2

        sum_centroids = centroids_reference + centroids

        identities = np.zeros((*left_shape, out_channels, out_channels))
        identities[..., np.arange(out_channels), np.arange(out_channels)] = 1

        return (
            identities
            + gamma[..., None, None] * sum_centroids[..., None, :] * sum_centroids[..., None]
            + 2 * centroids_reference[..., None] * centroids[..., None, :]
        )
