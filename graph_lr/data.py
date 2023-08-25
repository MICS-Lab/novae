import numpy as np
from anndata import AnnData
from torch.utils.data import Dataset

from .sampling import HopSampler


class StandardDataset(Dataset):
    def __init__(self, adata: AnnData, n_hops: int) -> None:
        self.adata = adata
        self.n_hops = n_hops

        self.sampler = HopSampler(self.adata)

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        return self.sampler.hop_travel(idx, self.n_hops)


class PairsDataset(Dataset):
    def __init__(
        self,
        adata: AnnData,
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

        self.sampler = HopSampler(self.adata)

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
