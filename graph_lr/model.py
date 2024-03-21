from __future__ import annotations

import logging

import lightning as L
import numpy as np
import torch
from anndata import AnnData
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ._constants import INT_CONF, REPR, SWAV_CLASSES
from .data import LocalAugmentationDataset
from .module import GenesEmbedding, GraphEncoder, SwavHead
from .utils import fill_invalid_indices, genes_union, prepare_adatas, tqdm

log = logging.getLogger(__name__)


class GraphLR(L.LightningModule):
    def __init__(
        self,
        adata: AnnData | list[AnnData],
        swav: bool,
        slide_key: str = None,
        embedding_size: int = 256,
        heads: int = 1,
        n_hops: int = 2,
        n_intermediate: int = 4,
        hidden_channels: int = 64,
        num_layers: int = 10,
        out_channels: int = 64,
        batch_size: int = 256,
        lr: float = 1e-3,
        temperature: float = 0.1,
        num_prototypes: int = 32,
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

        ### Losses
        self.bce_loss = nn.BCELoss()

        ### Dataset
        self.dataset = self.init_dataset()

    def __repr__(self) -> str:
        return f"GraphLR model with {self.n_obs} cells and {len(self.var_names)} genes"

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
            x_ngh = self.backbone.edge_x(batch[1])

            loss = self.bce_loss(x_main, torch.ones_like(x_main, device=x_main.device))
            loss += self.bce_loss(x_ngh, torch.zeros_like(x_ngh, device=x_ngh.device))

        self.log(
            "loss",
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
            slide_key=self.slide_key,
            batch_size=self.hparams.batch_size,
            n_hops=self.hparams.n_hops,
            n_intermediate=self.hparams.n_intermediate,
        )

    def train_dataloader(self):
        self.dataset.eval = False
        return DataLoader(
            self.dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )

    def test_dataloader(self, adata: AnnData | list[AnnData] | None = None):
        if adata is None:
            dataset = self.dataset
        else:
            dataset = self.init_dataset(adata)

        dataset.eval = True
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=False, drop_last=False
        )

    def on_train_epoch_start(self):
        self.swav_head.prototypes.requires_grad = self.current_epoch > 0

        self.dataset.shuffle_grouped_indices()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @torch.no_grad()
    def interaction_confidence(self) -> None:
        for adata in self.adatas:
            loader = self.test_dataloader(adata)

            out = torch.concatenate(
                [
                    self.backbone.edge_x(batch[0]) - self.backbone.edge_x(batch[1])
                    for batch in tqdm(loader)
                ]
            )

            adata.obs[INT_CONF] = fill_invalid_indices(out, adata, loader.dataset.valid_indices)

    @torch.no_grad()
    def swav_clusters(self, use_codes: bool = False) -> None:
        for adata in self.adatas:
            loader = self.test_dataloader(adata)

            out = []
            for data_main, *_ in tqdm(loader):
                x_main = self.backbone.node_x(data_main)
                out_ = F.normalize(x_main, dim=1, p=2)
                scores = out_ @ self.swav_head.prototypes

                out.append(scores if use_codes else scores.argmax(1))

            out = torch.cat(out)

            if use_codes:
                out = self.swav_head.sinkhorn(out)
                out = out.argmax(1)

            adata.obs[SWAV_CLASSES] = fill_invalid_indices(
                out, adata, loader.dataset.valid_indices, fill_value=np.nan
            )

    @torch.no_grad()
    def representations(self) -> None:
        for adata in self.adatas:
            loader = self.test_dataloader(adata)

            out = []
            for data_main, *_ in tqdm(loader):
                x_main = self.backbone.node_x(data_main)
                out_ = F.normalize(x_main, dim=1, p=2)
                out.append(out_)

            out = torch.concat(out, dim=0)
            adata.obsm[REPR] = fill_invalid_indices(out, adata, loader.dataset.valid_indices)
