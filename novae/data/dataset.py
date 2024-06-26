from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix, lil_matrix
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from .. import utils
from .._constants import Keys, Nums
from ..module import CellEmbedder
from . import AnnDataTorch


class NeighborhoodDataset(Dataset):
    """
    Dataset used for training and inference.

    It extracts the the neighborhood of a cell, and convert it to PyTorch Geometric Data.

    Attributes:
        valid_indices (list[np.ndarray]): List containing, for each `adata`, an array that denotes the indices of the cells whose neighborhood is valid.
        obs_ilocs (np.ndarray): An array of shape `(total_valid_indices, 2)`. The first column corresponds to the adata index, and the second column is the cell index for the corresponding adata.
        shuffled_obs_ilocs (np.ndarray): same as obs_ilocs, but shuffled. Each batch will contain cells from the same slide.
    """

    valid_indices: list[np.ndarray]
    obs_ilocs: np.ndarray
    shuffled_obs_ilocs: np.ndarray

    @utils.format_docs
    def __init__(
        self,
        adatas: list[AnnData],
        cell_embedder: CellEmbedder,
        batch_size: int,
        n_hops_local: int,
        n_hops_view: int,
    ) -> None:
        """_summary_

        Args:
            adatas: A list of `AnnData` objects.
            cell_embedder: A [novae.module.CellEmbedder][] object.
            batch_size: The model batch size.
            {n_hops_local}
            {n_hops_view}
        """
        super().__init__()
        self.adatas = adatas
        self.cell_embedder = cell_embedder
        self.anndata_torch = AnnDataTorch(self.adatas, self.cell_embedder)

        self.training = False

        self.batch_size = batch_size
        self.n_hops_local = n_hops_local
        self.n_hops_view = n_hops_view

        self.single_slide_mode = len(self.adatas) == 1 and len(np.unique(self.adatas[0].obs[Keys.SLIDE_ID])) == 1

        self._init_dataset()

    def _init_dataset(self):
        for adata in self.adatas:
            adjacency: csr_matrix = adata.obsp[Keys.ADJ]

            adata.obsp[Keys.ADJ_LOCAL] = _to_adjacency_local(adjacency, self.n_hops_local)
            adata.obsp[Keys.ADJ_PAIR] = _to_adjacency_pair(adjacency, self.n_hops_view)
            adata.obs[Keys.IS_VALID_OBS] = adata.obsp[Keys.ADJ_PAIR].sum(1).A1 > 0

        self.valid_indices = [np.where(adata.obs[Keys.IS_VALID_OBS])[0] for adata in self.adatas]

        self.obs_ilocs = None
        if self.single_slide_mode:
            self.obs_ilocs = np.array([(0, obs_index) for obs_index in self.valid_indices[0]])

        self.slides_metadata: pd.DataFrame = pd.concat(
            [
                self._adata_slides_metadata(adata_index, obs_indices)
                for adata_index, obs_indices in enumerate(self.valid_indices)
            ],
            axis=0,
        )

        self.shuffle_obs_ilocs()

    def __len__(self) -> int:
        if self.training:
            n_obs = len(self.shuffled_obs_ilocs)
            return min(n_obs, max(Nums.MIN_DATASET_LENGTH, int(n_obs * Nums.MAX_DATASET_LENGTH_RATIO)))

        assert self.obs_ilocs is not None, "Multi-adata mode not yet supported for inference"
        return len(self.obs_ilocs)

    def __getitem__(self, index: int) -> dict[str, Data]:
        """Gets a sample from the dataset, with one "main" graph and its corresponding "view" graph.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A dictionnary whose keys are "main" and "view", and values are PyTorch Geometric `Data` objects.
        """
        if self.training:
            adata_index, obs_index = self.shuffled_obs_ilocs[index]
        else:
            adata_index, obs_index = self.obs_ilocs[index]

        adjacency_pair: csr_matrix = self.adatas[adata_index].obsp[Keys.ADJ_PAIR]

        plausible_nghs = adjacency_pair[obs_index].indices
        ngh_index = np.random.choice(list(plausible_nghs), size=1)[0]

        data = self.to_pyg_data(adata_index, obs_index)
        data_ngh = self.to_pyg_data(adata_index, ngh_index)

        return {"main": data, "view": data_ngh}

    def to_pyg_data(self, adata_index: int, obs_index: int) -> Data:
        """Create a PyTorch Geometric Data object for the input cell

        Args:
            adata_index: The index of the `AnnData` object
            obs_index: The index of the input cell for the corresponding `AnnData` object

        Returns:
            A Data object
        """
        adjacency_local: csr_matrix = self.adatas[adata_index].obsp[Keys.ADJ_LOCAL]
        obs_indices = adjacency_local[obs_index].indices

        x, genes_indices = self.anndata_torch[adata_index, obs_indices]

        adjacency: csr_matrix = self.adatas[adata_index].obsp[Keys.ADJ]

        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency[obs_indices][:, obs_indices])
        edge_attr = edge_weight[:, None].to(torch.float32) / Nums.CELLS_CHARACTERISTIC_DISTANCE

        return Data(x=x, genes_indices=genes_indices, edge_index=edge_index, edge_attr=edge_attr)

    def _adata_slides_metadata(self, adata_index: int, obs_indices: list[int]) -> pd.DataFrame:
        obs_counts: pd.Series = self.adatas[adata_index].obs.iloc[obs_indices][Keys.SLIDE_ID].value_counts()
        slides_metadata = obs_counts.to_frame()
        slides_metadata[Keys.ADATA_INDEX] = adata_index
        slides_metadata[Keys.N_BATCHES] = (slides_metadata["count"] // self.batch_size).clip(1)
        return slides_metadata

    def shuffle_obs_ilocs(self):
        """Shuffle the indices of the cells to be used in the dataset (for training only)."""
        if self.single_slide_mode:
            self.shuffled_obs_ilocs = self.obs_ilocs[np.random.permutation(len(self.obs_ilocs))]
            return

        adata_indices = np.empty((0, self.batch_size), dtype=int)
        batched_obs_indices = np.empty((0, self.batch_size), dtype=int)

        for uid in self.slides_metadata.index:
            adata_index = self.slides_metadata.loc[uid, Keys.ADATA_INDEX]
            adata = self.adatas[adata_index]
            _obs_indices = np.where((adata.obs[Keys.SLIDE_ID] == uid) & adata.obs[Keys.IS_VALID_OBS])[0]
            _obs_indices = np.random.permutation(_obs_indices)

            n_elements = self.slides_metadata.loc[uid, Keys.N_BATCHES] * self.batch_size
            if len(_obs_indices) >= n_elements:
                _obs_indices = _obs_indices[:n_elements]
            else:
                _obs_indices = np.random.choice(_obs_indices, size=n_elements)

            _obs_indices = _obs_indices.reshape((-1, self.batch_size))

            adata_indices = np.concatenate([adata_indices, np.full_like(_obs_indices, adata_index)], axis=0)
            batched_obs_indices = np.concatenate([batched_obs_indices, _obs_indices], axis=0)

        permutation = np.random.permutation(len(batched_obs_indices))
        adata_indices = adata_indices[permutation].flatten()
        obs_indices = batched_obs_indices[permutation].flatten()

        self.shuffled_obs_ilocs = np.stack([adata_indices, obs_indices], axis=1)


def _to_adjacency_local(adjacency: csr_matrix, n_hops_local: int) -> csr_matrix:
    """
    Creates an adjancency matrix for which all nodes
    at a distance inferior to `n_hops_local` are linked.
    """
    assert n_hops_local >= 1, f"n_hops_local must be greater than 0. Found {n_hops_local}."

    adjacency_local: lil_matrix = adjacency.copy().tolil()
    adjacency_local.setdiag(1)
    for _ in range(n_hops_local - 1):
        adjacency_local = adjacency_local @ adjacency
    return adjacency_local.tocsr()


def _to_adjacency_pair(adjacency: csr_matrix, n_hops_view: int) -> csr_matrix:
    """
    Creates an adjacancy matrix for which all nodes separated by
    precisely `n_hops_view` nodes are linked.
    """
    assert n_hops_view >= 1, f"n_hops_view must be greater than 0. Found {n_hops_view}."

    if n_hops_view == 1:
        adjacency_pair = adjacency.copy()
        adjacency_pair.setdiag(0)
        adjacency_pair.eliminate_zeros()
        return adjacency_pair

    adjacency_pair = adjacency.copy()
    adjacency_pair.setdiag(1)
    for i in range(n_hops_view - 1):
        if i == n_hops_view - 2:
            adjacency_previous = adjacency_pair.copy()
        adjacency_pair = adjacency_pair @ adjacency
    adjacency_pair = adjacency_pair.tolil()
    adjacency_pair[adjacency_previous.nonzero()] = 0
    adjacency_pair: csr_matrix = adjacency_pair.tocsr()
    adjacency_pair.eliminate_zeros()
    return adjacency_pair
