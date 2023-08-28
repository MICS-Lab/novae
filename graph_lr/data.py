import numpy as np
import torch
from anndata import AnnData
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix


def _factor_augment(x: torch.Tensor) -> torch.Tensor:
    n_nodes = x.shape[0]
    factor = 1 + torch.tanh(torch.randn(n_nodes)) / 5
    return x * factor[:, None]


class HopSampler:
    def __init__(
        self, adata: AnnData, x: torch.Tensor, embedding, transform=None
    ) -> None:
        self.adata = adata
        self.x = x
        self.embedding = embedding
        self.transform = transform

        self.A = self.adata.obsp["spatial_connectivities"]

    def _to_pyg_graph(self, indices) -> Data:
        x = self.x[indices]
        if self.transform is not None:
            for t in self.transform:
                x = t(x)
        x = self.embedding(x)
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


class StandardDataset(Dataset):
    def __init__(self, adata: AnnData, x: torch.Tensor, embedding, n_hops: int) -> None:
        self.adata = adata
        self.n_hops = n_hops

        self.sampler = HopSampler(self.adata, x, embedding)

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        return self.sampler.hop_travel(idx, self.n_hops)


class PairsDataset(Dataset):
    def __init__(
        self,
        adata: AnnData,
        x: torch.Tensor,
        embedding,
        n_hops: int,
        n_intermediate: int = None,
        max_attemps: int = 10,
    ) -> None:
        self.adata = adata
        self.n_hops = n_hops
        self.n_intermediate = (
            2 * self.n_hops if n_intermediate is None else n_intermediate
        )
        self.max_attemps = max_attemps

        self.sampler = HopSampler(self.adata, x, embedding, transform=[_factor_augment])

    def __len__(self):
        return 10_000

    def __getitem__(self, idx):
        # TODO: remove origin_index from plausible if no pair_index possible

        for _ in range(self.max_attemps):
            origin_index = np.random.choice(self.adata.n_obs, size=1)[0]
            pair_index = self.sampler.get_pair_node(origin_index, self.n_intermediate)

            if pair_index is None:
                continue

            return self.sampler.hop_travel(
                origin_index, self.n_hops
            ), self.sampler.hop_travel(pair_index, self.n_hops)
