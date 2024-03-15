from __future__ import annotations

import importlib

import numpy as np
import pytorch_lightning as pl
import torch
from anndata import AnnData
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from ._constants import EPS
from .data import LocalAugmentationDataset
from .module import GenesEmbedding, GraphEncoder, SwavHead
from .utils import prepare_adatas


class GraphCL(pl.LightningModule):
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

        self.adatas = [adata] if isinstance(adata, AnnData) else adata
        self.slide_key = slide_key

        prepare_adatas(self.adatas)

        self.embedding = GenesEmbedding(adata.var_names, embedding_size)

        # PCA init embeddings (valid only if x centered)
        pca = PCA(n_components=embedding_size)
        pca.fit(self.x_numpy)  # TODO: fix it
        self.embedding.embedding.weight.data = torch.tensor(pca.components_.T)

        self.module = GraphEncoder(embedding_size, hidden_channels, num_layers, out_channels, heads)
        self.projection = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        self.classifier = nn.Linear(out_channels, 1)
        self.bce_loss = nn.BCELoss()

        self.swav_head = SwavHead(out_channels, num_prototypes, temperature)

    def forward(self, batch):
        return [self.module(view) for view in batch]

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
            self.embedding,
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
            self.embedding,
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
                self.module(batch[0])[1] - self.module(batch[1])[1]
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
            np_, _ = self.module(h1)
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
    def embeddings(self) -> np.ndarray:
        emb = []

        loader = self.test_dataloader()

        for h1, _, _ in loader:
            np_, _ = self.module(h1)
            out1 = F.normalize(np_, dim=1, p=2)
            emb.append(out1)

        embeddings = torch.concat(emb, dim=0).numpy(force=True)
        res = np.zeros((self.adata.n_obs, embeddings.shape[1]))
        res[loader.dataset.valid_indices] = embeddings

        return res

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
