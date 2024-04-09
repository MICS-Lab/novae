from __future__ import annotations

import logging

import lightning as L
import numpy as np
import torch
from anndata import AnnData
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ._constants import CODES, INT_CONF, REPR, SWAV_CLASSES
from .data import LocalAugmentationDataset
from .module import GenesEmbedding, GraphAugmentation, GraphEncoder, SwavHead
from .utils import (
    fill_edge_scores,
    fill_invalid_indices,
    genes_union,
    prepare_adatas,
    tqdm,
)

log = logging.getLogger(__name__)


class Novae(L.LightningModule):
    def __init__(
        self,
        adata: AnnData | list[AnnData],
        swav: bool = True,
        slide_key: str = None,
        embedding_size: int = 100,
        heads: int = 4,
        n_hops: int = 2,
        n_intermediate: int = 4,
        hidden_channels: int = 64,
        num_layers: int = 10,
        out_channels: int = 64,
        batch_size: int = 256,
        lr: float = 1e-3,
        temperature: float = 0.1,
        num_prototypes: int = 256,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["adata", "slide_key"])

        self.adatas = prepare_adatas(adata)
        self.slide_key = slide_key

        ### Embeddings
        self.genes_embedding = GenesEmbedding(self.var_names, embedding_size)
        # self.genes_embedding.pca_init(self.adatas) # TODO: fix it

        ### Modules
        self.backbone = GraphEncoder(
            embedding_size, hidden_channels, num_layers, out_channels, heads
        )
        self.swav_head = SwavHead(out_channels, num_prototypes, temperature)
        self.augmentation = GraphAugmentation(self.genes_embedding)

        ### Losses
        self.bce_loss = nn.BCELoss()

        ### Dataset
        self.dataset = self.init_dataset()

    def __repr__(self) -> str:
        return f"Novae model with {self.n_obs} cells and {len(self.var_names)} genes"

    @property
    def var_names(self) -> list[str]:
        return genes_union(self.adatas)

    @property
    def n_obs(self) -> int:
        return sum(adata.n_obs for adata in self.adatas)

    def training_step(self, batch: tuple[Data, Data, Data], batch_idx: int):
        if self.hparams.swav:
            x_main = self.backbone.node_x(batch[0])
            x_ngh = self.backbone.node_x(batch[2])

            loss = self.swav_head(x_main, x_ngh)
        else:
            x_main = self.backbone.edge_x(batch[0])
            x_shuffle = self.backbone.edge_x(batch[1])

            loss = self.bce_loss(x_main, torch.ones_like(x_main, device=x_main.device))
            loss += self.bce_loss(x_shuffle, torch.zeros_like(x_shuffle, device=x_shuffle.device))

        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=True,
            batch_size=self.hparams.batch_size,
            prog_bar=True,
        )

        return loss

    def init_dataset(self, adata: AnnData | list[AnnData] | None = None):
        if adata is None:
            adatas = self.adatas
        elif isinstance(adata, AnnData):
            adatas = [adata]
        elif isinstance(adata, list):
            adatas = adata
        else:
            raise ValueError(f"Invalid type {type(adata)} for argument adata")

        return LocalAugmentationDataset(
            adatas,
            self.genes_embedding,
            augmentation=self.augmentation,
            batch_size=self.hparams.batch_size,
            n_hops=self.hparams.n_hops,
            n_intermediate=self.hparams.n_intermediate,
        )

    def on_train_epoch_start(self):
        self.swav_head.prototypes.requires_grad = self.current_epoch > 0

        self.dataset.shuffle_obs_ilocs()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @torch.no_grad()
    def interaction_confidence(self, adata: AnnData | list[AnnData] | None = None) -> None:
        for adata in self.get_adatas(adata):
            loader = self.predict_dataloader(adata)

            out = torch.concatenate(
                [
                    self.backbone.edge_x(batch[0]) - self.backbone.edge_x(batch[1])
                    for batch in tqdm(loader)
                ]
            )

            adata.obs[INT_CONF] = fill_invalid_indices(out, adata, loader.dataset.valid_indices)

    @torch.no_grad()
    def swav_classes(self, adata: AnnData | list[AnnData] | None = None) -> None:
        for adata in self.get_adatas(adata):
            assert CODES in adata.obsm, f"Codes are not computed. Run model.codes() first."

            codes = adata.obsm[CODES]
            adata.obs[SWAV_CLASSES] = np.where(
                np.isnan(codes).any(1), np.nan, np.argmax(codes, 1).astype(object)
            )

    def predict_step(self, batch):
        data_main, *_ = batch
        x_main = self.backbone.node_x(data_main)
        out = F.normalize(x_main, dim=1, p=2)
        return out @ self.swav_head.prototypes

    def _get_trainer(self, trainer: L.Trainer | None) -> L.Trainer:
        if trainer is not None:
            return trainer
        return L.Trainer()

    @torch.no_grad()
    def codes(
        self,
        adata: AnnData | list[AnnData] | None = None,
        sinkhorn: bool = True,
        trainer: L.Trainer | None = None,
    ) -> None:
        for adata in self.get_adatas(adata):
            trainer = self._get_trainer(trainer)
            dataset = self.init_dataset(adata)

            out = []
            for data_main, *_ in tqdm(dataset.predict_dataloader()):
                x_main = self.backbone.node_x(data_main)
                out_ = F.normalize(x_main, dim=1, p=2)
                scores = out_ @ self.swav_head.prototypes

                out.append(scores)

            out = torch.cat(out)

            if sinkhorn:
                out = self.swav_head.sinkhorn(out)

            adata.obsm[CODES] = fill_invalid_indices(
                out, adata, dataset.valid_indices, fill_value=np.nan, dtype=np.float32
            )

    @torch.no_grad()
    def edge_scores(
        self, adata: AnnData | list[AnnData] | None = None, sinkhorn: bool = True
    ) -> None:
        for adata in self.get_adatas(adata):
            loader = self.predict_dataloader(adata)

            edge_scores = []
            for data_main, *_ in tqdm(loader):
                edge_scores += self.backbone.edge_x(data_main, return_weights=True)

            edge_scores = torch.cat(edge_scores)

            adata.obsm[CODES] = fill_edge_scores(
                edge_scores,
                adata,
                loader.dataset.valid_indices,
                fill_value=np.nan,
                dtype=np.float32,
            )

    @torch.no_grad()
    def representation(self, adata: AnnData | list[AnnData] | None = None) -> None:
        for adata in self.get_adatas(adata):
            loader = self.predict_dataloader(adata)

            out = []
            for data_main, *_ in tqdm(loader):
                x_main = self.backbone.node_x(data_main)
                out_ = F.normalize(x_main, dim=1, p=2)
                out.append(out_)

            out = torch.concat(out, dim=0)
            adata.obsm[REPR] = fill_invalid_indices(out, adata, loader.dataset.valid_indices)

    def get_adatas(self, adata: AnnData | list[AnnData] | None):
        if adata is None:
            return self.adatas
        return prepare_adatas(adata, vocabulary=self.genes_embedding.vocabulary)
