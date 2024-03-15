from __future__ import annotations

import importlib

import numpy as np
import pytorch_lightning as pl
import torch
from anndata import AnnData
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

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

    def forward(self, batch: list[Tensor]) -> list[Tensor]:
        return [self.backbone(view) for view in batch]

    def training_step(self, batch, batch_idx):
        (np, ep), (_, ep_shuffle), (np_ngh, _) = self(batch)

        if self.hparams.swav:
            loss = self.swav_head(np, np_ngh)
        else:
            loss = self.bce_loss(ep, torch.ones_like(ep, device=ep.device)) + self.bce_loss(
                ep_shuffle, torch.zeros_like(ep_shuffle, device=ep_shuffle.device)
            )

        self.log(
            "loss",
            loss,
            on_epoch=True,
            on_step=True,
            batch_size=self.hparams.batch_size,
            prog_bar=True,
        )

        return loss

    def train_dataloader(self):
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
        return DataLoader(
            self.dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )

    def test_dataloader(self):
        dataset = LocalAugmentationDataset(
            self.adatas,
            self.genes_embedding,
            slide_key=self.slide_key,
            batch_size=self.hparams.batch_size,
            n_hops=self.hparams.n_hops,
            n_intermediate=self.hparams.n_intermediate,
        )
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=False, drop_last=False
        )

    def on_train_epoch_start(self):
        self.swav_head.prototypes.requires_grad = self.current_epoch > 0

        self.dataset.shuffle_grouped_indices()

    @torch.no_grad()
    def delta(self) -> np.ndarray:
        if importlib.util.find_spec("ipywidgets") is not None:
            from tqdm.autonotebook import tqdm
        else:
            from tqdm import tqdm

        loader = self.test_dataloader()

        out = torch.concatenate(
            [
                self.backbone(batch[0])[1] - self.backbone(batch[1])[1]
                for batch in tqdm(loader, desc="DataLoader")
            ]
        )

        delta = np.zeros(self.adata.n_obs, dtype=float)
        delta[loader.dataset.valid_indices] = out.numpy(force=True)

        return delta

    @torch.no_grad()
    def swav_clusters(self, use_codes: bool = False) -> np.ndarray:
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
        res[loader.dataset.valid_indices] = preds.numpy(force=True).astype(str)

        return res

    @torch.no_grad()
    def representations(self) -> np.ndarray:
        emb = []

        loader = self.test_dataloader()

        for h1, _, _ in loader:
            np_, _ = self.backbone(h1)
            out1 = F.normalize(np_, dim=1, p=2)
            emb.append(out1)

        representations = torch.concat(emb, dim=0).numpy(force=True)
        res = np.zeros((self.adata.n_obs, representations.shape[1]))
        res[loader.dataset.valid_indices] = representations

        return res

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
