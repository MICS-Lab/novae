import importlib

import numpy as np
import pytorch_lightning as pl
import torch
from anndata import AnnData
from torch import nn, optim
from torch_geometric.loader import DataLoader

from .data import LocalAugmentationDataset
from .module import GenesEmbedding, GraphEncoder


class GraphCL(pl.LightningModule):
    def __init__(
        self,
        adata: AnnData,
        batch_key: str = None,
        # obsm_key="X_pca",
        embedding_size: int = 256,
        heads: int = 1,
        n_hops: int = 2,
        n_intermediate: int = 4,
        hidden_channels: int = 64,
        num_layers: int = 10,
        out_channels: int = 64,
        batch_size: int = 32,
        lr: float = 1e-3,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["adata"])

        self.adata = adata
        self.x_numpy = self.adata.X  # log1p expressions
        self.x = torch.tensor(self.x_numpy)

        self.embedding = GenesEmbedding(adata.var_names, embedding_size)
        self.module = GraphEncoder(
            embedding_size, hidden_channels, num_layers, out_channels, heads
        )
        self.projection = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        self.classifier = nn.Linear(out_channels, 1)
        self.bce_loss = nn.BCELoss()

    def forward(self, batch, edge_pooling: bool = False):
        data1, data2 = batch

        h1 = self.module(data1, edge_pooling=edge_pooling)
        h2 = self.module(data2, edge_pooling=edge_pooling)

        return h1, h2

    def training_step(self, batch, batch_idx):
        h1, h2 = self(batch)

        loss = self.bce_loss(h1, torch.ones_like(h1, device=h1.device)) + self.bce_loss(
            h2, torch.zeros_like(h2, device=h2.device)
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
        dataset = LocalAugmentationDataset(
            self.adata,
            self.x,
            self.embedding,
            delta_th=0.5,
            n_hops=self.hparams.n_hops,
            n_intermediate=self.hparams.n_intermediate,
        )
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )

    def test_dataloader(self):
        dataset = LocalAugmentationDataset(
            self.adata,
            self.x,
            self.embedding,
            n_hops=self.hparams.n_hops,
            n_intermediate=self.hparams.n_intermediate,
        )
        dataset
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=False, drop_last=False
        )

    @torch.no_grad()
    def delta(self) -> torch.Tensor:
        if importlib.util.find_spec("ipywidgets") is not None:
            from tqdm.autonotebook import tqdm
        else:
            from tqdm import tqdm

        loader = self.test_dataloader()

        out = torch.concatenate(
            [
                self.module(batch[0]) - self.module(batch[1])
                for batch in tqdm(loader, desc="DataLoader")
            ]
        )

        delta = np.zeros(self.adata.n_obs, dtype=float)
        delta[loader.dataset.valid_indices] = out.numpy(force=True)

        return delta

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @torch.no_grad()
    def sinkhorn(out, epsilon: float = 0.05, sinkhorn_iterations: int = 3):
        """Q is K-by-B for consistency with notations from the paper (out: B*K)"""
        Q = torch.exp(out / epsilon).t()
        Q /= torch.sum(Q)

        B = Q.shape[1]
        K = Q.shape[0]

        for _ in range(sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()
