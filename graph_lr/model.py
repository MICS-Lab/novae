import importlib

import pytorch_lightning as pl
import torch
from anndata import AnnData
from torch import nn, optim
from torch_geometric.loader import DataLoader

from .data import HopSampler, PairsDataset, StandardDataset
from .module import ContrastiveLoss, Embedding, GraphEncoder


class GraphCL(pl.LightningModule):
    def __init__(
        self,
        adata: AnnData,
        # obsm_key="X_pca",
        embedding_size: int = 64,
        n_hops: int = 2,
        n_intermediate: int = 4,
        hidden_channels: int = 64,
        num_layers: int = 5,
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

        self.n_genes = self.x.shape[1]

        self.embedding = Embedding(self.n_genes, embedding_size)
        self.module = GraphEncoder(
            embedding_size, hidden_channels, num_layers, out_channels
        )
        self.projection = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        self.contrastive_loss = ContrastiveLoss(self.hparams.batch_size, temperature)

    def forward(self, batch):
        data1, data2 = batch

        h1, h2 = self.module(data1), self.module(data2)

        return h1, h2

    def training_step(self, batch, batch_idx):
        h1, h2 = self(batch)
        loss = self.contrastive_loss(h1, h2)

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
        dataset = PairsDataset(
            self.adata,
            self.x,
            self.embedding,
            self.hparams.n_hops,
            n_intermediate=self.hparams.n_intermediate,
        )
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )

    @torch.no_grad()
    def embeddings(self):
        if importlib.util.find_spec("ipywidgets") is not None:
            from tqdm.autonotebook import tqdm
        else:
            from tqdm import tqdm

        dataset = StandardDataset(self.adata, self.x, self.embedding, self.hparams.n_hops)
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)
        return torch.concatenate(
            [self.module(batch) for batch in tqdm(loader, desc="DataLoader")]
        )

    def baseline_embeddings(self):
        sampler = HopSampler(self.adata, self.x)

        import numpy as np

        return np.stack(
            [
                self.x_numpy[
                    sampler.hop_travel(i, self.hparams.n_hops, as_pyg=False)
                ].mean(0)
                for i in range(self.adata.n_obs)
            ]
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
