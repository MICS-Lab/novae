import logging
import math

import anndata
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from sklearn.cluster import AgglomerativeClustering, KMeans
from torch import Tensor, nn

from .. import utils
from .._constants import Nums

log = logging.getLogger(__name__)


class SwavHead(L.LightningModule):
    unshared_prototypes: nn.ParameterDict | None

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

        self._shared_prototypes = nn.Parameter(torch.empty((self.num_prototypes, self.output_size)))
        self._shared_prototypes = nn.init.kaiming_uniform_(self._shared_prototypes, a=math.sqrt(5), mode="fan_out")
        self.normalize_prototypes(self._shared_prototypes)
        self.unshared_prototypes = None

        self.reset_clustering()

    def init_unshared_prototypes(self, slide_ids: list[str], unshared_prototypes_ratio: float) -> None:
        n_unshared_prototypes = int(self.num_prototypes * unshared_prototypes_ratio)

        self.unshared_prototypes = nn.ParameterDict(
            {
                str(slide_id): nn.Parameter(torch.empty((n_unshared_prototypes, self.output_size)))
                for slide_id in slide_ids
            }
        )

        for slide_id in slide_ids:
            self.unshared_prototypes[str(slide_id)] = nn.init.kaiming_uniform_(
                self.unshared_prototypes[str(slide_id)], a=math.sqrt(5), mode="fan_out"
            )
            self.normalize_prototypes(self.unshared_prototypes[str(slide_id)])

    @torch.no_grad()
    def normalize_prototypes(self, prototypes: Tensor):
        prototypes.data = F.normalize(prototypes.data, dim=1, p=2)

    def forward(self, z1: Tensor, z2: Tensor, slide_id: str | None) -> tuple[Tensor, Tensor]:
        """Compute the SwAV loss for two batches of neighborhood graph views.

        Args:
            z1: Batch containing graphs representations `(B, output_size)`
            z2: Batch containing graphs representations `(B, output_size)`

        Returns:
            The SwAV loss, and the mean entropy normalized (for monitoring).
        """
        self.normalize_prototypes(self._shared_prototypes)
        if slide_id is not None:
            self.normalize_prototypes(self.unshared_prototypes[str(slide_id)])

        projections1 = self.projection(z1, slide_id)  # (B, K)
        projections2 = self.projection(z2, slide_id)  # (B, K)

        q1 = self.sinkhorn(projections1)  # (B, K)
        q2 = self.sinkhorn(projections2)  # (B, K)

        loss = -0.5 * (self.cross_entropy_loss(q1, projections2) + self.cross_entropy_loss(q2, projections1))

        return loss, _mean_entropy_normalized(q1)

    def cross_entropy_loss(self, q: Tensor, p: Tensor) -> Tensor:
        return torch.mean(torch.sum(q * F.log_softmax(p / self.temperature, dim=1), dim=1))

    def projection(self, z: Tensor, slide_id: str | None = None) -> Tensor:
        """Compute the projection of the (normalized) representations over the prototypes.

        Args:
            z: The representations of one batch, of size `(B, O)`.

        Returns:
            The projections of size `(B, K)`.
        """
        if slide_id is not None:
            prototypes = self._shared_prototypes
            prototypes = torch.cat([prototypes, self.unshared_prototypes[str(slide_id)]])
        else:
            prototypes = self.prototypes

        z_normalized = F.normalize(z, dim=1, p=2)
        return z_normalized @ prototypes.T

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
        if self.mode.zero_shot:
            return self._kmeans_prototypes
        else:
            return torch.cat([self._shared_prototypes] + list(self.unshared_prototypes.values()))

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

    def _adata_prototypes(self) -> AnnData:
        def _to_adata(prototypes: Tensor, name: str = "shared") -> AnnData:
            adata = AnnData(X=prototypes.numpy(force=True))
            adata.obs["name"] = name
            return adata

        return anndata.concat(
            [_to_adata(self._shared_prototypes)]
            + [_to_adata(protos, name=name) for name, protos in self.unshared_prototypes.items()],
            index_unique="-",
        )


@torch.no_grad()
def _mean_entropy_normalized(q: Tensor) -> Tensor:
    entropy = -(q * torch.log2(q + Nums.EPS)).sum(-1)
    max_entropy = torch.log2(torch.tensor(q.shape[-1]))
    return (entropy / max_entropy).mean()
