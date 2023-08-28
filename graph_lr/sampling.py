import numpy as np
import torch
from anndata import AnnData
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix


class HopSampler:
    def __init__(self, adata: AnnData, x: torch.Tensor) -> None:
        self.adata = adata
        self.x = x
        self.A = self.adata.obsp["spatial_connectivities"]

    def _to_pyg_graph(self, indices) -> Data:
        x = self.x[indices]
        edge_index, _ = from_scipy_sparse_matrix(self.A[indices][:, indices])

        return Data(x=x, edge_index=edge_index)

    def get_pair_node(self, origin_index: int, n_intermediate: int):
        # TODO: travel on similar cells: use PCA distance?
        visited = {origin_index}
        for _ in range(n_intermediate):
            visited |= set(self.A[list(visited)].indices)

        plausible = set(self.A[list(visited)].indices) - visited

        return np.random.choice(list(plausible), size=1)[0] if plausible else None

    def hop_travel(self, index: int, n_hops: int, as_pyg: bool = True):
        indices = {index}

        for _ in range(n_hops):
            indices = indices | set(self.A[list(indices)].indices)

        return self._to_pyg_graph(list(indices)) if as_pyg else list(indices)
