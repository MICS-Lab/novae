import logging

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix, lil_matrix
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from .. import settings, utils
from .._constants import Keys, Nums
from ..module import CellEmbedder
from . import get_torch_converter

log = logging.getLogger(__name__)


class NovaeDataset(Dataset):
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

    def __init__(
        self,
        adatas: list[AnnData],
        cell_embedder: CellEmbedder | str,
        batch_size: int,
        n_hops_local: int,
        n_hops_view: int,
        sample_cells: int | None = None,
    ) -> None:
        """NovaeDataset constructor.

        Args:
            adatas: A list of `AnnData` objects.
            cell_embedder: A [novae.module.CellEmbedder][] object or a string specifying the embedding name.
            batch_size: The model batch size.
            n_hops_local: Number of hops between a cell and its neighborhood cells.
            n_hops_view: Number of hops between a cell and the origin of a second graph (or 'view').
            sample_cells: If not None, the dataset if used to sample the subgraphs from precisely `sample_cells` cells.
        """
        super().__init__()
        self.adatas = adatas
        self.cell_embedder = cell_embedder
        self.torch_converter = get_torch_converter(self.adatas, self.cell_embedder)

        self.training = False

        self.batch_size = batch_size
        self.n_hops_local = n_hops_local
        self.n_hops_view = n_hops_view
        self.sample_cells = sample_cells

        self.single_adata = len(self.adatas) == 1
        self.single_slide_mode = self.single_adata and len(np.unique(self.adatas[0].obs[Keys.SLIDE_ID])) == 1

        self._init_dataset()

    def __repr__(self) -> str:
        multi_slide_mode, multi_adata = not self.single_slide_mode, not self.single_adata
        n_samples = sum(len(indices) for indices in self.valid_indices)
        return f"{self.__class__.__name__} with {n_samples} samples ({multi_slide_mode=}, {multi_adata=})"

    def _init_dataset(self):
        for adata in self.adatas:
            adjacency: csr_matrix = adata.obsp[Keys.ADJ]

            if Keys.NOVAE_UNS not in adata.uns:
                adata.uns[Keys.NOVAE_UNS] = {}

            if Keys.ADJ_LOCAL not in adata.obsp or adata.uns[Keys.NOVAE_UNS].get("n_hops_local") != self.n_hops_local:
                adata.obsp[Keys.ADJ_LOCAL] = _to_adjacency_local(adjacency, self.n_hops_local)
                adata.uns[Keys.NOVAE_UNS]["n_hops_local"] = self.n_hops_local

            if Keys.ADJ_VIEW not in adata.obsp or adata.uns[Keys.NOVAE_UNS].get("n_hops_view") != self.n_hops_view:
                adata.obsp[Keys.ADJ_VIEW] = _to_adjacency_view(adjacency, self.n_hops_view)
                adata.uns[Keys.NOVAE_UNS]["n_hops_view"] = self.n_hops_view

            adata.obs[Keys.IS_VALID_OBS] = adata.obsp[Keys.ADJ_VIEW].sum(1).A1 > 0

        ratio_valid_obs = pd.concat([adata.obs[Keys.IS_VALID_OBS] for adata in self.adatas]).mean()
        if ratio_valid_obs < Nums.RATIO_VALID_CELLS_TH:
            log.warning(
                f"Only {ratio_valid_obs:.2%} of the cells have a valid neighborhood.\n"
                "Consider running `novae.spatial_neighbors` with a larger `radius`."
            )

        self.valid_indices = [utils.valid_indices(adata) for adata in self.adatas]

        self.obs_ilocs = None
        if self.single_adata:
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
        if self.sample_cells is not None:
            return min(self.sample_cells, len(self.shuffled_obs_ilocs))

        if self.training:
            n_obs = len(self.shuffled_obs_ilocs)
            return min(n_obs, max(Nums.MIN_DATASET_LENGTH, int(n_obs * Nums.MAX_DATASET_LENGTH_RATIO)))

        assert self.single_adata, "Multi-adata mode not supported for inference"

        return len(self.obs_ilocs)

    def __getitem__(self, index: int) -> dict[str, Data]:
        """Gets a sample from the dataset, with one "main" graph and its corresponding "view" graph (only during training).

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A dictionnary whose keys are names, and values are PyTorch Geometric `Data` objects. The `"view"` graph is only provided during training.
        """
        if self.training or self.sample_cells is not None:
            adata_index, obs_index = self.shuffled_obs_ilocs[index]
        else:
            adata_index, obs_index = self.obs_ilocs[index]

        data = self.to_pyg_data(adata_index, obs_index)

        if not self.training:
            return {"main": data}

        adjacency_pair: csr_matrix = self.adatas[adata_index].obsp[Keys.ADJ_VIEW]
        cell_view_index = np.random.choice(list(adjacency_pair[obs_index].indices), size=1)[0]

        return {"main": data, "view": self.to_pyg_data(adata_index, cell_view_index)}

    def to_pyg_data(self, adata_index: int, obs_index: int) -> Data:
        """Create a PyTorch Geometric Data object for the input cell

        Args:
            adata_index: The index of the `AnnData` object
            obs_index: The index of the input cell for the corresponding `AnnData` object

        Returns:
            A Data object
        """
        adata = self.adatas[adata_index]
        adjacency_local: csr_matrix = adata.obsp[Keys.ADJ_LOCAL]
        obs_indices = adjacency_local[obs_index].indices

        adjacency: csr_matrix = adata.obsp[Keys.ADJ]
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency[obs_indices][:, obs_indices])
        edge_attr = (
            edge_weight[:, None].to(torch.float32) * settings.scale_to_microns / Nums.CELLS_CHARACTERISTIC_DISTANCE
        )

        histo_embeddings = None
        if Keys.HISTO_EMBEDDINGS in adata.obsm and not settings.disable_multimodal:
            histo_embeddings = adata.obsm[Keys.HISTO_EMBEDDINGS][[obs_index]]
            histo_embeddings = torch.tensor(histo_embeddings, dtype=torch.float32)

        x, genes_indices = self.torch_converter[adata_index, obs_indices]

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            genes_indices=genes_indices,
            slide_id=adata.obs[Keys.SLIDE_ID].iloc[obs_index],
            histo_embeddings=histo_embeddings,
        )

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

    def _adata_slides_metadata(self, adata_index: int, obs_indices: list[int]) -> pd.DataFrame:
        obs_counts: pd.Series = self.adatas[adata_index].obs.iloc[obs_indices][Keys.SLIDE_ID].value_counts()
        slides_metadata = obs_counts.to_frame()
        slides_metadata[Keys.ADATA_INDEX] = adata_index
        slides_metadata[Keys.N_BATCHES] = (slides_metadata["count"] // self.batch_size).clip(1)
        return slides_metadata


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


def _to_adjacency_view(adjacency: csr_matrix, n_hops_view: int) -> csr_matrix:
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
    adjacency_pair: csr_matrix = adjacency_pair.tocsr()  # type: ignore[no-redef]
    adjacency_pair.eliminate_zeros()
    return adjacency_pair
