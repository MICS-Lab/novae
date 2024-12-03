import numpy as np
import torch
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from .._constants import Keys, Nums
from ..module import CellEmbedder
from ..utils import sparse_std


class AnnDataTorch:
    tensors: list[Tensor] | None
    genes_indices_list: list[Tensor]

    def __init__(self, adatas: list[AnnData], cell_embedder: CellEmbedder):
        """Converting AnnData objects to PyTorch tensors.

        Args:
            adatas: A list of `AnnData` objects.
            cell_embedder: A [novae.module.CellEmbedder][] object.
        """
        super().__init__()
        self.adatas = adatas
        self.cell_embedder = cell_embedder

        self.genes_indices_list = [self._adata_to_genes_indices(adata) for adata in self.adatas]
        self.tensors = None

        self.means, self.stds, self.label_encoder = self._compute_means_stds()

        # Tensors are loaded in memory for low numbers of cells
        if sum(adata.n_obs for adata in self.adatas) < Nums.N_OBS_THRESHOLD:
            self.tensors = [self.to_tensor(adata) for adata in self.adatas]

    def _adata_to_genes_indices(self, adata: AnnData) -> Tensor:
        return self.cell_embedder.genes_to_indices(adata.var_names[self._keep_var(adata)])[None, :]

    def _keep_var(self, adata: AnnData) -> AnnData:
        return adata.var[Keys.USE_GENE]

    def _compute_means_stds(self) -> tuple[Tensor, Tensor, LabelEncoder]:
        means, stds = {}, {}

        for adata in self.adatas:
            slide_ids = adata.obs[Keys.SLIDE_ID]
            for slide_id in slide_ids.cat.categories:
                adata_slide = adata[adata.obs[Keys.SLIDE_ID] == slide_id, self._keep_var(adata)]

                mean = adata_slide.X.mean(0)
                mean = mean.A1 if isinstance(mean, np.matrix) else mean
                means[slide_id] = mean.astype(np.float32)

                std = adata_slide.X.std(0) if isinstance(adata_slide.X, np.ndarray) else sparse_std(adata_slide.X, 0).A1
                stds[slide_id] = std.astype(np.float32)

        label_encoder = LabelEncoder()
        label_encoder.fit(list(means.keys()))

        means = [torch.tensor(means[slide_id]) for slide_id in label_encoder.classes_]
        stds = [torch.tensor(stds[slide_id]) for slide_id in label_encoder.classes_]

        return means, stds, label_encoder

    def to_tensor(self, adata: AnnData) -> Tensor:
        """Get the normalized gene expressions of the cells in the dataset.
        Only the genes of interest are kept (known genes and highly variable).

        Args:
            adata: An `AnnData` object.

        Returns:
            A `Tensor` containing the normalized gene expresions.
        """
        adata = adata[:, self._keep_var(adata)]

        if len(np.unique(adata.obs[Keys.SLIDE_ID])) == 1:
            slide_id_index = self.label_encoder.transform([adata.obs.iloc[0][Keys.SLIDE_ID]])[0]
            mean, std = self.means[slide_id_index], self.stds[slide_id_index]
        else:
            slide_id_indices = self.label_encoder.transform(adata.obs[Keys.SLIDE_ID])
            mean = torch.stack([self.means[i] for i in slide_id_indices])  # TODO: avoid stack (only if not fast enough)
            std = torch.stack([self.stds[i] for i in slide_id_indices])

        X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        X = torch.tensor(X, dtype=torch.float32)
        X = (X - mean) / (std + Nums.EPS)

        return X

    def __getitem__(self, item: tuple[int, slice]) -> tuple[Tensor, Tensor]:
        """Get the expression values for a subset of cells (corresponding to a subgraph).

        Args:
            item: A `tuple` containing the index of the `AnnData` object and the indices of the cells in the neighborhoods.

        Returns:
            A `Tensor` of normalized gene expressions and a `Tensor` of gene indices.
        """
        adata_index, obs_indices = item

        if self.tensors is not None:
            return self.tensors[adata_index][obs_indices], self.genes_indices_list[adata_index]

        adata = self.adatas[adata_index]
        adata_view = adata[obs_indices]

        return self.to_tensor(adata_view), self.genes_indices_list[adata_index]
