from __future__ import annotations

import importlib

import numpy as np
import pytorch_lightning as pl
import torch
from anndata import AnnData
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ._constants import REPR, SWAV_CLASSES
from .data import LocalAugmentationDataset
from .module import GenesEmbedding, GraphEncoder, SwavHead
from .utils import genes_union, prepare_adatas


class GraphLR(pl.LightningModule):
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
        self.dataset = self.init_dataset()

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

    @property
    def var_names(self) -> list[str]:
        return genes_union(self.adatas)

    def n_obs(self) -> int:
        return sum(adata.n_obs for adata in self.adatas)

    def forward(self, batch: tuple[Data, Data, Data]) -> tuple[Data, Data, Data]:
        return [self.backbone(view) for view in batch]

    def training_step(self, batch: tuple[Data, Data, Data], batch_idx: int):
        if self.hparams.swav:
            x_main = self.backbone.node_x(batch[0])
            x_ngh = self.backbone.node_x(batch[2])

            loss = self.swav_head(x_main, x_ngh)
        else:
            x_main = self.backbone.edge_x(batch[0])
            x_ngh = self.backbone.edge_x(batch[1])

            loss = self.bce_loss(
                x_main, torch.ones_like(x_main, device=x_main.device)
            ) + self.bce_loss(x_ngh, torch.zeros_like(x_ngh, device=x_ngh.device))

        self.log(
            "loss",
            loss,
            on_epoch=True,
            on_step=True,
            batch_size=self.hparams.batch_size,
            prog_bar=True,
        )

        return loss

    def init_dataset(self, adata: AnnData | list[AnnData] | None):
        if adata is None:
            adata = self.adatas
        elif isinstance(adata, AnnData):
            adata = [adata]
        else:
            assert isinstance(adata, list), f"Invalid `adata` argument of type {type(adata)}"

        self.dataset = LocalAugmentationDataset(
            self.adatas,
            self.genes_embedding,
            eval=False,
            slide_key=self.slide_key,
            batch_size=self.hparams.batch_size,
            delta_th=0.5,
            n_hops=self.hparams.n_hops,
            n_intermediate=self.hparams.n_intermediate,
        )

    def train_dataloader(self):
        self.dataset.eval = False
        return DataLoader(
            self.dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )

    def test_dataloader(self):
        if importlib.util.find_spec("ipywidgets") is not None:
            from tqdm.autonotebook import tqdm
        else:
            from tqdm import tqdm

        self.dataset.eval = True
        loader = DataLoader(
            self.dataset, batch_size=self.hparams.batch_size, shuffle=False, drop_last=False
        )
        return tqdm(loader, desc="DataLoader")

    def on_train_epoch_start(self):
        self.swav_head.prototypes.requires_grad = self.current_epoch > 0

        self.dataset.shuffle_grouped_indices()

    @torch.no_grad()
    def interaction_confidence(self) -> np.ndarray:
        out = torch.concatenate(
            [
                self.backbone(batch[0])[1] - self.backbone(batch[1])[1]
                for batch in self.test_dataloader()
            ]
        )

        delta = np.zeros(self.adata.n_obs, dtype=float)
        delta[self.dataset.valid_indices] = out.numpy(force=True)

        return delta

    @torch.no_grad()
    def swav_clusters(self, adata: AnnData | None = None, use_codes: bool = False) -> None:
        preds = []

        loader = self.test_dataloader()

        for h1, _, _ in loader:
            np_, _ = self.backbone(h1)
            out1 = F.normalize(np_, dim=1, p=2)
            scores1 = out1 @ self.swav_head.prototypes

            if use_codes:
                preds.append(scores1)
            else:
                pred = scores1.argmax(1)
                preds.append(pred)

        preds = torch.cat(preds)

        if use_codes:
            preds = self.swav_head.sinkhorn(preds)
            preds = preds.argmax(1)

        res = np.full(self.adata.n_obs, "nan")
        res[self.dataset.valid_indices] = preds.numpy(force=True).astype(str)

        adata.obsm[SWAV_CLASSES] = res

    @torch.no_grad()
    def representations(self, adata: AnnData | None = None) -> None:
        emb = []

        loader = self.test_dataloader()

        for h1, _, _ in loader:
            np_, _ = self.backbone(h1)
            out1 = F.normalize(np_, dim=1, p=2)
            emb.append(out1)

        representations = torch.concat(emb, dim=0).numpy(force=True)
        res = np.zeros((self.adata.n_obs, representations.shape[1]))
        res[loader.dataset.valid_indices] = representations

        adata.obsm[REPR] = res

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
