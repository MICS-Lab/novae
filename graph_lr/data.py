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
        all_nodes: bool = False,
        n_hops: int = 2,
        n_intermediate: int = None,
    ) -> None:
        self.adata = adata
        self.x = x
        self.embedding = embedding
        self.all_nodes = all_nodes
        self.n_hops = n_hops
        self.n_intermediate = n_intermediate or 2 * self.n_hops

        self.genes_indices = self.embedding.genes_to_indices(adata.var_names)

        self.A = adata.obsp["spatial_connectivities"]
        self.valid_indices = np.where(self.A.sum(1).A1 > 0)[0]

    def __len__(self) -> int:
        if self.all_nodes:
            return self.adata.n_obs
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if not self.all_nodes:
            idx = self.valid_indices[idx]

        return self.hop_travel(idx, n_hops=self.n_hops, shuffle_pair=True)

    def _to_pyg_graph(
        self, indices, shuffle_pair: bool = False
    ) -> Union[Data, Tuple[Data, Data]]:
        x = self.x[indices]

        x = self.embedding(x, self.genes_indices)
        edge_index, _ = from_scipy_sparse_matrix(self.A[indices][:, indices])

        data = Data(x=x, edge_index=edge_index)

        if shuffle_pair:
            shuffled_x = x[torch.randperm(x.size()[0])]
            shuffled_data = Data(x=shuffled_x, edge_index=edge_index)
            return data, shuffled_data

        return data

    def hop_travel(
        self, index: int, n_hops: int, as_pyg: bool = True, shuffle_pair: bool = False
    ):
        indices = {index}

        for _ in range(n_hops):
            indices = indices | set(self.A[list(indices)].indices)

        indices = list(indices)

        return (
            self._to_pyg_graph(indices, shuffle_pair=shuffle_pair) if as_pyg else indices
        )

    # def get_pair_node(self, origin_index: int, n_intermediate: int):
    #     # TODO: travel on similar cells: use PCA distance?
    #     visited = {origin_index}
    #     for _ in range(n_intermediate):
    #         visited |= set(self.A[list(visited)].indices)

    #     plausible = set(self.A[list(visited)].indices) - visited

    #     return np.random.choice(list(plausible), size=1)[0] if plausible else None
