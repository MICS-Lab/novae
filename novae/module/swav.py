import logging
import math

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, KMeans
from torch import Tensor, nn

from .. import utils
from .._constants import Nums

log = logging.getLogger(__name__)


class SwavHead(L.LightningModule):
    queue: None | Tensor  # (n_slides, QUEUE_SIZE, num_prototypes)

    def __init__(
        self,
        mode: utils.Mode,
        output_size: int,
        num_prototypes: int,
        temperature: float,
    ):
        """SwavHead module, adapted from the paper ["Unsupervised Learning of Visual Features by Contrasting Cluster Assignments"](https://arxiv.org/abs/2006.09882).

        Args:
            output_size: Size of the representations, i.e. the encoder outputs (`O` in the article).
            num_prototypes: Number of prototypes (`K` in the article).
            temperature: Temperature used in the cross-entropy loss.
        """
        super().__init__()
        self.mode = mode
        self.output_size = output_size
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        self._prototypes = nn.Parameter(torch.empty((self.num_prototypes, self.output_size)))
        self._prototypes = nn.init.kaiming_uniform_(self._prototypes, a=math.sqrt(5), mode="fan_out")
        self.normalize_prototypes()
        self.min_prototypes = 0

        self.queue = None

        self.reset_clustering()

    def set_min_prototypes(self, min_prototypes_ratio: float):
        self.min_prototypes = int(self.num_prototypes * min_prototypes_ratio)

    def init_queue(self, slide_ids: list[str]) -> None:
        """Initialize the slide-queue.

        Args:
            slide_ids: A list of slide ids.
        """
        del self.queue

        shape = (len(slide_ids), Nums.QUEUE_SIZE, self.num_prototypes)
        self.register_buffer("queue", torch.full(shape, 1 / self.num_prototypes))

        self.slide_label_encoder = {slide_id: i for i, slide_id in enumerate(slide_ids)}

    @torch.no_grad()
    def normalize_prototypes(self):
        self.prototypes.data = F.normalize(self.prototypes.data, dim=1, p=2)

    def forward(self, z1: Tensor, z2: Tensor, slide_id: str | None) -> tuple[Tensor, Tensor]:
        """Compute the SwAV loss for two batches of neighborhood graph views.

        Args:
            z1: Batch containing graphs representations `(B, output_size)`
            z2: Batch containing graphs representations `(B, output_size)`

        Returns:
            The SwAV loss, and the mean entropy normalized (for monitoring).
        """
        self.normalize_prototypes()

        projections1 = self.projection(z1)  # (B, K)
        projections2 = self.projection(z2)  # (B, K)

        ilocs = self.prototype_ilocs(projections1, slide_id)

        projections1, projections2 = projections1[:, ilocs], projections2[:, ilocs]

        q1 = self.sinkhorn(projections1)  # (B, K) or (B, len(ilocs))
        q2 = self.sinkhorn(projections2)  # (B, K) or (B, len(ilocs))

        loss = -0.5 * (self.cross_entropy_loss(q1, projections2) + self.cross_entropy_loss(q2, projections1))

        return loss, _mean_entropy_normalized(q1)

    def cross_entropy_loss(self, q: Tensor, p: Tensor) -> Tensor:
        return torch.mean(torch.sum(q * F.log_softmax(p / self.temperature, dim=1), dim=1))

    def projection(self, z: Tensor) -> Tensor:
        """Compute the projection of the (normalized) representations over the prototypes.

        Args:
            z: The representations of one batch, of size `(B, O)`.

        Returns:
            The projections of size `(B, K)`.
        """
        z_normalized = F.normalize(z, dim=1, p=2)
        return z_normalized @ self.prototypes.T

    @torch.no_grad()
    def prototype_ilocs(self, projections: Tensor, slide_id: str | None = None) -> Tensor:
        """Get the indices of the prototypes to use for the current slide.

        Args:
            projections: Projections of the (normalized) representations over the prototypes, of size `(B, K)`.
            slide_id: ID of the slide, or `None`.

        Returns:
            The indices of the prototypes to use, or an `Ellipsis` if all prototypes.
        """
        if (self.queue is None) or (slide_id is None) or self.mode.zero_shot:
            return ...

        slide_index = self.slide_label_encoder[slide_id]

        self.queue[slide_index, 1:] = self.queue[slide_index, :-1].clone()
        self.queue[slide_index, 0] = projections.topk(3, dim=0).values[-1]  # top3 more robust than max

        weights, thresholds = self.queue_weights()
        slide_weights = weights[slide_index]

        ilocs = torch.where(slide_weights >= thresholds)[0]

        if len(ilocs) >= self.min_prototypes:
            return ilocs

        other_locs = torch.where(slide_weights < thresholds)[0]
        other_locs = other_locs[torch.topk(slide_weights[other_locs], self.min_prototypes - len(ilocs)).indices]

        return torch.cat([ilocs, other_locs])

    def queue_weights(self) -> tuple[Tensor, Tensor]:
        """Convert the queue to a matrix of prototype weight per slide.

        Returns:
            A tensor of shape `(n_slides, K)`, and a tensor of shape (K,).
        """
        max_projections = self.queue.max(dim=1).values

        thresholds = max_projections.max(0).values * Nums.QUEUE_WEIGHT_THRESHOLD_RATIO
        thresholds -= 1 - Nums.QUEUE_WEIGHT_THRESHOLD_RATIO  # ensure that for max-weights < 0 are above the threshold

        return max_projections, thresholds

    @torch.no_grad()
    def sinkhorn(self, projections: Tensor) -> Tensor:
        """Apply the Sinkhorn-Knopp algorithm to the projections.

        Args:
            projections: Projections of the (normalized) representations over the prototypes, of size `(B, K)`.

        Returns:
            The soft codes from the Sinkhorn-Knopp algorithm, with shape `(B, K)`.
        """
        Q = torch.exp(projections / Nums.SWAV_EPSILON)  # (B, K)
        Q /= torch.sum(Q)

        B, K = Q.shape

        for _ in range(Nums.SINKHORN_ITERATIONS):
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        return Q / Q.sum(dim=1, keepdim=True)  # ensure rows sum to 1 (for cross-entropy loss)

    def set_kmeans_prototypes(self, latent: np.ndarray):
        assert (
            len(latent) >= self.num_prototypes
        ), f"The number of valid cells ({len(latent)}) must be greater than the number of prototypes ({self.num_prototypes})."

        kmeans = KMeans(n_clusters=self.num_prototypes, random_state=0, n_init="auto")
        X = latent / (Nums.EPS + np.linalg.norm(latent, axis=1)[:, None])

        kmeans_prototypes = kmeans.fit(X).cluster_centers_
        kmeans_prototypes = kmeans_prototypes / (Nums.EPS + np.linalg.norm(kmeans_prototypes, axis=1)[:, None])

        self._kmeans_prototypes = torch.nn.Parameter(torch.tensor(kmeans_prototypes))

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

    def reset_clustering(self, only_zero_shot: bool = False) -> None:
        attrs = self.mode.zero_shot_clustering_attrs if only_zero_shot else self.mode.all_clustering_attrs
        for attr in attrs:
            setattr(self, attr, None)

    def set_clustering(self, clustering: None, clusters_levels: None) -> None:
        setattr(self, self.mode.clustering_attr, clustering)
        setattr(self, self.mode.clusters_levels_attr, clusters_levels)

    def hierarchical_clustering(self) -> None:
        """
        Perform hierarchical clustering on the prototypes. Saves the full tree of clusters.
        """
        X = self.prototypes.data.numpy(force=True)  # (K, O)

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

    def map_leaves_domains(self, series: pd.Series, level: int) -> pd.Series:
        """Map leaves to the parent domain from the corresponding level of the hierarchical tree.

        Args:
            series: Leaves classes
            level: Level of the hierarchical clustering tree (or, number of clusters)

        Returns:
            Series of classes.
        """
        return series.map(lambda x: f"D{self.clusters_levels[-level, int(x[1:])]}" if isinstance(x, str) else x)

    def find_level(self, leaves_indices: np.ndarray, n_domains: int):
        sub_clusters_levels = self.clusters_levels[:, leaves_indices]
        for level in range(1, self.num_prototypes):
            _n_domains = len(np.unique(sub_clusters_levels[-level]))
            if _n_domains == n_domains:
                return level
        raise ValueError(f"Could not find a level with {n_domains=}")


@torch.no_grad()
def _mean_entropy_normalized(q: Tensor) -> Tensor:
    entropy = -(q * torch.log2(q + Nums.EPS)).sum(-1)
    max_entropy = torch.log2(torch.tensor(q.shape[-1]))
    return (entropy / max_entropy).mean()
