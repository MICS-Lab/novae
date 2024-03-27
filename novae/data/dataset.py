from __future__ import annotations

import lightning as L
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix, lil_matrix
from torch.distributions import Exponential
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from .._constants import (
    ADATA_INDEX_KEY,
    ADJ,
    ADJ_LOCAL,
    ADJ_PAIR,
    IS_VALID_KEY,
    N_BATCHES,
    SLIDE_KEY,
)
from ..module import GenesEmbedding
from .convert import AnnDataTorch


class LocalAugmentationDataset(L.LightningDataModule):
    valid_indices: list[np.ndarray]
    obs_ilocs: list[tuple[int, int]]

    def __init__(
        self,
        adatas: list[AnnData],
        genes_embedding: GenesEmbedding,
        panel_dropout: float = 0.4,
        gene_expression_dropout: float = 0.1,
        background_noise_lambda: float = 5.0,
        sensitivity_noise_std: float = 0.05,
        slide_key: bool = None,
        batch_size: int = None,
        n_hops: int = 2,
        n_intermediate: int = 4,
    ) -> None:
        super().__init__()
        self.adatas = adatas
        self.anndata_torch = AnnDataTorch(self.adatas)

        self.genes_embedding = genes_embedding
        self.slide_key = slide_key
        self.eval = False

        self.save_hyperparameters(ignore=["adata", "genes_embedding", "slide_key"])

        self.init_dataset()

        self.background_noise_distribution = Exponential(torch.tensor(background_noise_lambda))

    def init_dataset(self):
        for adata in self.adatas:
            adjacency: csr_matrix = adata.obsp[ADJ]

            adata.obsp[ADJ_LOCAL] = _to_adjacency_local(adjacency, self.hparams.n_hops)
            adata.obsp[ADJ_PAIR] = _to_adjacency_pair(adjacency, self.hparams.n_intermediate)
            adata.obs[IS_VALID_KEY] = adata.obsp[ADJ_PAIR].sum(1).A1 > 0

        self.valid_indices = [np.where(adata.obs[IS_VALID_KEY])[0] for adata in self.adatas]
        self.init_obs_ilocs()

    def init_obs_ilocs(self):
        if self.slide_key is None:
            self.obs_ilocs = [
                (adata_index, obs_index)
                for adata_index, obs_indices in enumerate(self.valid_indices)
                for obs_index in obs_indices
            ]
            return

        assert (
            self.hparams.batch_size is not None
        ), "When using `slide_key`, you need to also provide `batch_size`"

        self.slides_metadata: pd.DataFrame = pd.concat(
            [
                self._adata_slides_metadata(adata_index, obs_indices)
                for adata_index, obs_indices in enumerate(self.valid_indices)
            ],
            axis=0,
        )
        self.shuffle_grouped_indices()

    def _adata_slides_metadata(self, adata_index: int, obs_indices: list[int]) -> pd.DataFrame:
        obs_counts: pd.Series = self.adatas[adata_index].obs[SLIDE_KEY][obs_indices].value_counts()
        slides_metadata = obs_counts.to_frame()
        slides_metadata[ADATA_INDEX_KEY] = adata_index
        slides_metadata[N_BATCHES] = (slides_metadata["count"] // self.hparams.batch_size).clip(1)
        return slides_metadata

    def shuffle_grouped_indices(self):
        if self.slide_key is None:
            return

        adata_indices = []
        batched_obs_indices = np.empty((0, self.hparams.batch_size), dtype=int)

        for uid in self.slides_metadata.index:
            adata_index = self.slides_metadata.loc[uid, ADATA_INDEX_KEY]
            adata = self.adatas[adata_index]
            _obs_indices = np.where((adata.obs[SLIDE_KEY] == uid) & adata.obs[IS_VALID_KEY])[0]
            _obs_indices = np.random.permutation(_obs_indices)

            n_elements = self.slides_metadata.loc[uid, N_BATCHES] * self.hparams.batch_size
            if len(_obs_indices) >= n_elements:
                _obs_indices = _obs_indices[:n_elements]
            else:
                _obs_indices = np.random.choice(_obs_indices, size=n_elements)

            _obs_indices = _obs_indices.reshape((-1, self.hparams.batch_size))

            adata_indices += [adata_index] * len(_obs_indices)
            batched_obs_indices = np.concatenate([batched_obs_indices, _obs_indices], axis=0)

        permutation = np.random.permutation(range(len(batched_obs_indices)))
        obs_indices = batched_obs_indices[permutation].flatten()
        self.obs_ilocs = list(zip(adata_indices, obs_indices))

        # TODO: remove?
        self.obs_ilocs = self.obs_ilocs[: self.hparams.batch_size * 200]

    def __len__(self) -> int:
        return len(self.obs_ilocs)

    def __getitem__(self, index: int) -> tuple[Data, Data, Data]:
        adata_index, obs_index = self.obs_ilocs[index]

        adjacency_pair: csr_matrix = self.adatas[adata_index].obsp[ADJ_PAIR]

        plausible_nghs = adjacency_pair[obs_index].indices
        ngh_index = np.random.choice(list(plausible_nghs), size=1)[0]

        data, data_shuffled = self.hop_travel(adata_index, obs_index, shuffle_pair=True)
        data_ngh = self.hop_travel(adata_index, ngh_index)

        return data, data_shuffled, data_ngh

    def hop_travel(
        self, adata_index: int, obs_index: int, shuffle_pair: bool = False
    ) -> Data | tuple[Data, Data]:
        adjacency: csr_matrix = self.adatas[adata_index].obsp[ADJ]
        adjacency_local: csr_matrix = self.adatas[adata_index].obsp[ADJ_LOCAL]

        indices = adjacency_local[obs_index].indices

        x, var_names = self.anndata_torch[adata_index, indices]

        x = self.transform(x, var_names)

        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency[indices][:, indices])
        edge_attr = edge_weight[:, None].to(torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if shuffle_pair:
            shuffled_x = x[torch.randperm(x.size()[0])]
            shuffled_data = Data(x=shuffled_x, edge_index=edge_index, edge_attr=edge_attr)
            return data, shuffled_data

        return data

    def transform(self, x: torch.Tensor, var_names: list[str]) -> torch.Tensor:
        genes_indices = self.genes_embedding.genes_to_indices(var_names)

        if self.eval:
            return self.genes_embedding(x, genes_indices)

        # noise background + sensitivity
        addition = self.background_noise_distribution.sample(sample_shape=(x.shape[1],))
        factor = (1 + torch.randn(x.shape[1]) * self.hparams.sensitivity_noise_std).clip(0, 2)
        x = x * factor + addition

        # gene expression dropout (= low quality gene)
        # indices = torch.randperm(x.shape[1])[: int(x.shape[1] * self.gene_expression_dropout)]
        # x[:, indices] = 0

        # gene subset (= panel change)
        n_vars = len(genes_indices)
        gene_subset_indices = torch.randperm(n_vars)[
            : int(n_vars * (1 - self.hparams.panel_dropout))
        ]

        x = self.genes_embedding(x[:, gene_subset_indices], genes_indices[gene_subset_indices])

        return x


def _to_adjacency_local(adjacency: csr_matrix, n_hops: int) -> csr_matrix:
    adjacency_local: lil_matrix = adjacency.copy().tolil()
    adjacency_local.setdiag(1)
    for _ in range(n_hops - 1):
        adjacency_local = adjacency_local @ adjacency
    return adjacency_local.tocsr()


def _to_adjacency_pair(adjacency: csr_matrix, n_intermediate: int) -> csr_matrix:
    adjacency_pair: lil_matrix = adjacency.copy().tolil()
    adjacency_pair.setdiag(1)
    for i in range(n_intermediate):
        if i == n_intermediate - 1:
            adjacency_previous: lil_matrix = adjacency_pair.copy()
        adjacency_pair = adjacency_pair @ adjacency
    adjacency_pair[adjacency_previous.nonzero()] = 0
    adjacency_pair: csr_matrix = adjacency_pair.tocsr()
    adjacency_pair.eliminate_zeros()
    return adjacency_pair
