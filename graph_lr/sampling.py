import numpy as np
import torch
from anndata import AnnData
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix


class HopSampler:
    def __init__(self, adata: AnnData) -> None:
        self.adata = adata
        self.A = self.adata.obsp["spatial_connectivities"]

    def _to_pyg_graph(self, indices, obsm_key: str = "X_pca"):
        x = self.adata.obsm[obsm_key][indices]
        edge_index, _ = from_scipy_sparse_matrix(self.A[indices][:, indices])

        return Data(x=torch.tensor(x), edge_index=edge_index)

    def get_pair_node(self, origin_index, n_intermediate):
        # TODO: travel on similar cells: use PCA distance?
        visited = {origin_index}
        for _ in range(n_intermediate):
            visited |= set(self.A[list(visited)].indices)

        plausible = set(self.A[list(visited)].indices) - visited

        return np.random.choice(list(plausible), size=1)[0] if plausible else None

    def hop_travel(self, index, n_hops):
        indices = {index}

        for _ in range(n_hops):
            indices = indices | set(self.A[list(indices)].indices)

        return self._to_pyg_graph(list(indices))
