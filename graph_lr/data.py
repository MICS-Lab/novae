from typing import Tuple, Union

import numpy as np
import torch
from anndata import AnnData
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from .module import GenesEmbedding


class LocalAugmentationDataset(Dataset):
    def __init__(
        self,
        adata: AnnData,
        x: torch.Tensor,
        embedding: GenesEmbedding,
        delta_th: float = None,
        n_hops: int = 2,
        n_intermediate: int = None,
    ) -> None:
        self.adata = adata
        self.x = x
        self.embedding = embedding
        self.delta_th = delta_th
        self.n_hops = n_hops
        self.n_intermediate = n_intermediate or 2 * self.n_hops

        self.genes_indices = self.embedding.genes_to_indices(adata.var_names)

        self.A = adata.obsp["spatial_connectivities"]

        self.valid_indices = [i for i in range(adata.n_obs) if self.node_is_valid(i)]

    def node_is_valid(self, index: int):
        if "delta" in self.adata.obs and self.delta_th is not None:
            if self.adata.obs["delta"].values[index] <= self.delta_th:
                return False

        return len(self.n_hop_neighbours(index)) > 0

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, dataset_index):
        index = self.valid_indices[dataset_index]

        plausible_nghs = self.n_hop_neighbours(index)
        ngh_index = np.random.choice(list(plausible_nghs), size=1)[0]

        data, data_shuffled = self.hop_travel(index, shuffle_pair=True)
        data_ngh = self.hop_travel(ngh_index)

        return data, data_shuffled, data_ngh

    def hop_travel(self, index: int, shuffle_pair: bool = False):
        indices = {index}

        for _ in range(self.n_hops):
            indices = indices | set(self.A[list(indices)].indices)

        indices = list(indices)

        x = self.x[indices]
        x = self.embedding(x, self.genes_indices)
        edge_index, _ = from_scipy_sparse_matrix(self.A[indices][:, indices])

        data = Data(x=x, edge_index=edge_index)

        if shuffle_pair:
            shuffled_x = x[torch.randperm(x.size()[0])]
            shuffled_data = Data(x=shuffled_x, edge_index=edge_index)
            return data, shuffled_data

        return data

    def n_hop_neighbours(self, origin_index: int) -> set:
        visited = {origin_index}

        for _ in range(self.n_intermediate):
            visited |= set(self.A[list(visited)].indices)

        return set(self.A[list(visited)].indices) - visited
