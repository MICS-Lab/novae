from __future__ import annotations

import lightning as L
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix, lil_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
from ..module import GenesEmbedding, GraphAugmentation
from .convert import AnnDataTorch


class LocalAugmentationDatamodule(L.LightningDataModule):
    valid_indices: list[np.ndarray]
    obs_ilocs: list[tuple[int, int]]

    def __init__(
        self,
        adatas: list[AnnData],
        genes_embedding: GenesEmbedding,
        augmentation: GraphAugmentation,
        batch_size: int,
        n_hops: int = 2,
        n_intermediate: int = 4,
    ) -> None:
        super().__init__()
        self.adatas = adatas
        self.anndata_torch = AnnDataTorch(self.adatas)

        self.genes_embedding = genes_embedding
        self.augmentation = augmentation
        self.transform = False

        self.batch_size = batch_size
        self.n_hops = n_hops
        self.n_intermediate = n_intermediate

        self.init_dataset()

    def init_dataset(self):
        for adata in self.adatas:
            adjacency: csr_matrix = adata.obsp[ADJ]

            adata.obsp[ADJ_LOCAL] = _to_adjacency_local(adjacency, self.n_hops)
            adata.obsp[ADJ_PAIR] = _to_adjacency_pair(adjacency, self.n_intermediate)
            adata.obs[IS_VALID_KEY] = adata.obsp[ADJ_PAIR].sum(1).A1 > 0

        self.valid_indices = [np.where(adata.obs[IS_VALID_KEY])[0] for adata in self.adatas]
        self.init_obs_ilocs()

    def init_obs_ilocs(self):
        if len(self.adatas) == 1:
            self.obs_ilocs = np.array([(0, obs_index) for obs_index in self.valid_indices[0]])
        else:
            self.obs_ilocs = None
            self.slides_metadata: pd.DataFrame = pd.concat(
                [
                    self._adata_slides_metadata(adata_index, obs_indices)
                    for adata_index, obs_indices in enumerate(self.valid_indices)
                ],
                axis=0,
            )

        self.shuffle_obs_ilocs()

    def shuffle_obs_ilocs(self):
        if len(self.adatas) == 1:
            self.shuffled_obs_ilocs = self.obs_ilocs[np.random.permutation(len(self.obs_ilocs))]
        else:
            self.shuffled_obs_ilocs = self.shuffle_grouped_indices()

    def _adata_slides_metadata(self, adata_index: int, obs_indices: list[int]) -> pd.DataFrame:
        obs_counts: pd.Series = (
            self.adatas[adata_index].obs.iloc[obs_indices][SLIDE_KEY].value_counts()
        )
        slides_metadata = obs_counts.to_frame()
        slides_metadata[ADATA_INDEX_KEY] = adata_index
        slides_metadata[N_BATCHES] = (slides_metadata["count"] // self.batch_size).clip(1)
        return slides_metadata

    def shuffle_grouped_indices(self):
        adata_indices = np.empty((0, self.batch_size), dtype=int)
        batched_obs_indices = np.empty((0, self.batch_size), dtype=int)

        for uid in self.slides_metadata.index:
            adata_index = self.slides_metadata.loc[uid, ADATA_INDEX_KEY]
            adata = self.adatas[adata_index]
            _obs_indices = np.where((adata.obs[SLIDE_KEY] == uid) & adata.obs[IS_VALID_KEY])[0]
            _obs_indices = np.random.permutation(_obs_indices)

            n_elements = self.slides_metadata.loc[uid, N_BATCHES] * self.batch_size
            if len(_obs_indices) >= n_elements:
                _obs_indices = _obs_indices[:n_elements]
            else:
                _obs_indices = np.random.choice(_obs_indices, size=n_elements)

            _obs_indices = _obs_indices.reshape((-1, self.batch_size))

            adata_indices = np.concatenate(
                [adata_indices, np.full_like(_obs_indices, adata_index)], axis=0
            )
            batched_obs_indices = np.concatenate([batched_obs_indices, _obs_indices], axis=0)

        permutation = np.random.permutation(len(batched_obs_indices))
        adata_indices = adata_indices[permutation].flatten()
        obs_indices = batched_obs_indices[permutation].flatten()
        shuffled_obs_ilocs = np.stack([adata_indices, obs_indices], axis=1)

        # TODO: remove cropped batch?
        return shuffled_obs_ilocs[: self.batch_size * 200]

    def __len__(self) -> int:
        if self.transform:
            return len(self.shuffled_obs_ilocs)
        assert self.obs_ilocs is not None, "Multi-adata mode not yet supported for inference"
        return len(self.obs_ilocs)

    def __getitem__(self, index: int) -> tuple[Data, Data, Data]:
        if self.transform:
            adata_index, obs_index = self.shuffled_obs_ilocs[index]
        else:
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

        x = self.augmentation(x, var_names, ignore=not self.transform)

        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency[indices][:, indices])
        edge_attr = edge_weight[:, None].to(torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if shuffle_pair:
            shuffled_x = x[torch.randperm(x.size()[0])]
            shuffled_data = Data(x=shuffled_x, edge_index=edge_index, edge_attr=edge_attr)
            return data, shuffled_data

        return data

    def train_dataloader(self) -> np.Any:
        self.transform = True
        return DataLoader(self, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def predict_dataloader(self):
        self.transform = False
        return DataLoader(self, batch_size=self.batch_size, shuffle=False, drop_last=False)


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
