from __future__ import annotations

import logging
import math

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, KMeans
from torch import Tensor, nn

from .._constants import Nums
from ..utils import Mode

log = logging.getLogger(__name__)


class SwavHead(L.LightningModule):
    queue: None | Tensor  # (n_slides, queue_size, num_prototypes)

    def __init__(
        self,
        mode: Mode,
        output_size: int,
        num_prototypes: int,
        temperature: float,
        temperature_weight_proto: float,
        min_prototypes_ratio: float,
    ):
        """SwavHead module, adapted from the paper "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments".

        Args:
            output_size: Size of the encoder embeddings.
            num_prototypes: Number of prototypes.
            temperature: Temperature used in the cross-entropy loss.
        """
        super().__init__()
        self.mode = mode
        self.output_size = output_size
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.temperature_weight_proto = temperature_weight_proto

        self._prototypes = nn.Parameter(torch.empty((self.num_prototypes, self.output_size)))
        self._prototypes = nn.init.kaiming_uniform_(self._prototypes, a=math.sqrt(5), mode="fan_out")
        self.normalize_prototypes()
        self.min_prototypes = int(num_prototypes * min_prototypes_ratio)

        self.queue = None

        self.reset_clustering()

    def init_queue(self, slide_ids: list[str]) -> None:
        del self.queue

        shape = (len(slide_ids), Nums.QUEUE_SIZE, self.num_prototypes)
        self.register_buffer("queue", torch.full(shape, 1 / self.num_prototypes))

        self.slide_label_encoder = {slide_id: i for i, slide_id in enumerate(slide_ids)}

    @torch.no_grad()
    def normalize_prototypes(self):
        self.prototypes.data = F.normalize(self.prototypes.data, dim=1, p=2)

    def forward(self, out1: Tensor, out2: Tensor, slide_id: str | None) -> tuple[Tensor, Tensor]:
        """Compute the SWAV loss for two batches of neighborhood graph views.

        Args:
            out1: Batch containing graphs representations `(B, output_size)`
            out2: Batch containing graphs representations `(B, output_size)`

        Returns:
            The SWAV loss, and the mean entropy normalized (for monitoring).
        """
        self.normalize_prototypes()

        scores1 = self.compute_scores(out1)  # (B, num_prototypes)
        scores2 = self.compute_scores(out2)  # (B, num_prototypes)

        ilocs = self.get_prototype_ilocs(scores1, slide_id)

        scores1, scores2 = scores1[:, ilocs], scores2[:, ilocs]

        q1 = self.sinkhorn(scores1)  # (B, num_prototypes) or (B, len(ilocs))
        q2 = self.sinkhorn(scores2)  # (B, num_prototypes) or (B, len(ilocs))

        loss = -0.5 * (self.cross_entropy_loss(q1, scores2) + self.cross_entropy_loss(q2, scores1))

        return loss, _mean_entropy_normalized(q1)

    def compute_scores(self, out: Tensor) -> Tensor:
        out = F.normalize(out, dim=1, p=2)
        return out @ self.prototypes.T

    @torch.no_grad()
    def get_prototype_ilocs(self, scores: Tensor, slide_id: str | None = None) -> Tensor:
        if slide_id is None or self.mode.zero_shot or not self.mode.queue_mode:
            return ...

        slide_index = self.slide_label_encoder[slide_id]
        slide_weights = F.softmax(scores / self.temperature_weight_proto, dim=1).mean(0)

        self.queue[slide_index, 1:] = self.queue[slide_index, :-1].clone()
        self.queue[slide_index, 0] = slide_weights

        if not self.mode.use_queue:
            return ...

        weights = self.get_queue_weights()[slide_index]
        ilocs = torch.where(weights >= Nums.QUEUE_WEIGHT_THRESHOLD)[0]

        return ilocs if len(ilocs) >= self.min_prototypes else torch.topk(weights, self.min_prototypes).indices

    def get_queue_weights(self):
        return self.sinkhorn(self.queue.mean(dim=1)) * self.num_prototypes

    def set_kmeans_prototypes(self, latent: np.ndarray):
        kmeans = KMeans(n_clusters=self.num_prototypes, random_state=0, n_init="auto")
        X = latent / (Nums.EPS + np.linalg.norm(latent, axis=1)[:, None])

        kmeans_prototypes = kmeans.fit(X).cluster_centers_
        kmeans_prototypes = kmeans_prototypes / (Nums.EPS + np.linalg.norm(kmeans_prototypes, axis=1)[:, None])

        self._kmeans_prototypes = torch.nn.Parameter(torch.tensor(kmeans_prototypes))

    @torch.no_grad()
    def sinkhorn(self, scores: Tensor) -> Tensor:
        """Apply the Sinkhorn-Knopp algorithm to the scores.

        Args:
            scores: The normalized embeddings projected into the prototypes, denoted Z@C.T in the paper.

        Returns:
            The soft codes from the Sinkhorn-Knopp algorithm, with shape `(B, num_prototypes)`.
        """
        Q = torch.exp(scores / Nums.SWAV_EPSILON)  # (B, num_prototypes)
        Q /= torch.sum(Q)

        B, num_prototypes = Q.shape

        for _ in range(Nums.SINKHORN_ITERATIONS):
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= num_prototypes
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        return Q / Q.sum(dim=1, keepdim=True)  # ensure rows sum to 1 (for cross-entropy loss)

    def cross_entropy_loss(self, q: Tensor, p: Tensor) -> Tensor:
        return torch.mean(torch.sum(q * F.log_softmax(p / self.temperature, dim=1), dim=1))

    def hierarchical_clustering(self) -> None:
        """
        Perform hierarchical clustering on the prototypes. Saves the full tree of clusters.
        """
        X = self.prototypes.data.numpy(force=True)  # (num_prototypes, output_size)

        _clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            compute_full_tree=True,
            metric="cosine",
            linkage="average",
        )
        _clustering.fit(X)

        _clusters_levels = np.zeros((len(X), len(X)), dtype=np.uint16)
        _clusters_levels[0] = np.arange(len(X))

        for i, (a, b) in enumerate(_clustering.children_):
            clusters = _clusters_levels[i]
            _clusters_levels[i + 1] = clusters
            _clusters_levels[i + 1, np.where((clusters == a) | (clusters == b))] = len(X) + i

        self.set_clustering(_clustering, _clusters_levels)

    @property
    def prototypes(self) -> nn.Parameter:
        return self._kmeans_prototypes if self.mode.zero_shot else self._prototypes

    @property
    def clustering(self) -> AgglomerativeClustering:
        clustering_attr = self.mode.clustering_attr

        if getattr(self, clustering_attr) is None:
            self.hierarchical_clustering()

        return getattr(self, clustering_attr)

    @property
    def clusters_levels(self) -> np.ndarray:
        clusters_levels_attr = self.mode.clusters_levels_attr

        if getattr(self, clusters_levels_attr) is None:
            self.hierarchical_clustering()

        return getattr(self, clusters_levels_attr)

    def map_leaves_domains(self, series: pd.Series, level: int) -> pd.Series:
        """Map leaves to the parent domain from the corresponding level of the hierarchical tree.

        Args:
            series: Leaves classes
            level: Level of the hierarchical clustering tree (or, number of clusters)

        Returns:
            Series of classes (one among `n_classes`).
        """
        return series.map(lambda x: f"N{self.clusters_levels[-level, int(x[1:])]}" if isinstance(x, str) else x)

    def find_level(self, leaves_indices: np.ndarray, n_domains: int):
        sub_clusters_levels = self.clusters_levels[:, leaves_indices]
        for level in range(1, self.num_prototypes):
            _n_domains = len(np.unique(sub_clusters_levels[-level]))
            if _n_domains == n_domains:
                return level
        raise ValueError(f"Could not find a level with {n_domains=}")

    def reset_clustering(self) -> None:
        for attr in self.mode.all_clustering_attrs:
            setattr(self, attr, None)

    def set_clustering(self, clustering: None, clusters_levels: None) -> None:
        setattr(self, self.mode.clustering_attr, clustering)
        setattr(self, self.mode.clusters_levels_attr, clusters_levels)


@torch.no_grad()
def _mean_entropy_normalized(q: Tensor) -> Tensor:
    entropy = -(q * torch.log2(q + Nums.EPS)).sum(-1)
    max_entropy = torch.log2(torch.tensor(q.shape[-1]))
    return (entropy / max_entropy).mean()
