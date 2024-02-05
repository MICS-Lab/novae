from typing import Tuple, Union

import numpy as np
import torch
from anndata import AnnData
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from .module import GenesEmbedding

IS_VALID_KEY = "is_valid"


class LocalAugmentationDataset(Dataset):
    def __init__(
        self,
        adata: AnnData,
        x: torch.Tensor,
        embedding: GenesEmbedding,
        slide_key: bool = None,
        batch_size: int = None,
        delta_th: float = None,
        n_hops: int = 2,
        n_intermediate: int = None,
    ) -> None:
        self.adata = adata
        self.x = x
        self.embedding = embedding

        self.slide_key = slide_key
        self.batch_size = batch_size

        self.delta_th = delta_th
        self.n_hops = n_hops
        self.n_intermediate = n_intermediate or 2 * self.n_hops

        self.genes_indices = self.embedding.genes_to_indices(adata.var_names)

        self.A = adata.obsp["spatial_connectivities"]

        self.A_local = self.A.copy()
        for _ in range(self.n_hops - 1):
            self.A_local = self.A_local @ self.A

        self.A_pair = self.A.copy()
        for i in range(self.n_intermediate):
            if i == self.n_intermediate - 1:
                A_previous = self.A_pair.copy()
            self.A_pair = self.A_pair @ self.A

        self.A_pair[A_previous.nonzero()] = 0
        self.A_pair.eliminate_zeros()

        self.adata.obs[IS_VALID_KEY] = self.A_pair.sum(1).A1 > 0
        self.valid_indices = np.where(self.adata.obs[IS_VALID_KEY])[0]

        self.init_obs_indices()

    def init_obs_indices(self):
        if self.slide_key is None:
            self.obs_indices = self.valid_indices
            return

        assert (
            self.batch_size is not None
        ), "When using `slide_key`, you need to also provide `batch_size`"

        self.batch_counts = self.adata.obs[self.slide_key][self.valid_indices].value_counts()
        self.batches_per_slide = (self.batch_counts // self.batch_size).clip(1)
        self.slides = self.batch_counts.index
        self.shuffle_grouped_indices()

    def shuffle_grouped_indices(self):
        if self.slide_key is None:
            return

        obs_indices = np.empty((0, self.batch_size), dtype=int)

        for slide in self.slides:
            indices = np.where(
                (self.adata.obs[self.slide_key] == slide) & self.adata.obs[IS_VALID_KEY]
            )[0]
            indices = np.random.permutation(indices)

            n_elements = self.batches_per_slide[slide] * self.batch_size
            if len(indices) >= n_elements:
                indices = indices[:n_elements]
            else:
                indices = np.random.choice(indices, size=n_elements)

            indices = indices.reshape((-1, self.batch_size))

            obs_indices = np.concatenate([obs_indices, indices], axis=0)

        np.random.shuffle(obs_indices)
        self.obs_indices = obs_indices.flatten()

        # TODO: remove
        self.obs_indices = self.obs_indices[: self.batch_size * 200]

    def __len__(self) -> int:
        return len(self.obs_indices)

    def __getitem__(self, dataset_index):
        index = self.obs_indices[dataset_index]

        plausible_nghs = self.A_pair[index].indices
        ngh_index = np.random.choice(list(plausible_nghs), size=1)[0]

        data, data_shuffled = self.hop_travel(index, shuffle_pair=True)
        data_ngh = self.hop_travel(ngh_index)

        return data, data_shuffled, data_ngh

    def hop_travel(self, index: int, shuffle_pair: bool = False):
        indices = self.A_local[index].indices

        x = self.x[indices]
        x = self.embedding(x, self.genes_indices)
        edge_index, _ = from_scipy_sparse_matrix(self.A[indices][:, indices])

        data = Data(x=x, edge_index=edge_index)

        if shuffle_pair:
            shuffled_x = x[torch.randperm(x.size()[0])]
            shuffled_data = Data(x=shuffled_x, edge_index=edge_index)
            return data, shuffled_data

        return data
