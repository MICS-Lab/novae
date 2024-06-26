from __future__ import annotations

import lightning as L
from anndata import AnnData
from torch_geometric.loader import DataLoader

from ..module import CellEmbedder
from . import NeighborhoodDataset


class NovaeDatamodule(L.LightningDataModule):
    """
    Datamodule used for training and inference. Small wrapper around the [novae.data.NeighborhoodDataset][]
    """

    def __init__(
        self,
        adatas: list[AnnData],
        cell_embedder: CellEmbedder,
        batch_size: int,
        n_hops_local: int,
        n_hops_view: int,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.dataset = NeighborhoodDataset(adatas, cell_embedder, batch_size, n_hops_local, n_hops_view)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """Get a Pytorch dataloader for prediction.

        Returns:
            The training dataloader.
        """
        self.dataset.training = True
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """Get a Pytorch dataloader for prediction or inference.

        Returns:
            The prediction dataloader.
        """
        self.dataset.training = False
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
