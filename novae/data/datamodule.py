from __future__ import annotations

import lightning as L
import numpy as np
from anndata import AnnData
from torch_geometric.loader import DataLoader

from ..module import GenesEmbedding
from .dataset import NeighborhoodDataset


class NovaeDatamodule(L.LightningDataModule):
    """
    Datamodule used for training and inference. Small wrapper around the `LocalAugmentationDataset`
    """

    def __init__(
        self,
        adatas: list[AnnData],
        genes_embedding: GenesEmbedding,
        batch_size: int,
        n_hops_local: int,
        n_hops_ngh: int,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.dataset = NeighborhoodDataset(adatas, genes_embedding, batch_size, n_hops_local, n_hops_ngh)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> np.Any:
        self.dataset.training = True
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        self.dataset.training = False
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
