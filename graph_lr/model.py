import importlib

import pytorch_lightning as pl
import torch
from anndata import AnnData
from torch import nn, optim
from torch_geometric.loader import DataLoader

from .data import HopSampler, ShuffledDataset, StandardDataset
from .module import ContrastiveLoss, GenesEmbedding, GraphEncoder


class GraphCL(pl.LightningModule):
    def __init__(
        self,
        adata: AnnData,
        batch_key: str = None,
        # obsm_key="X_pca",
        embedding_size: int = 64,
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

        self.sigmoid = nn.Sigmoid()

        self.classifier = nn.Linear(out_channels, 1)
        self.bce_loss = nn.BCELoss()
        # self.contrastive_loss = ContrastiveLoss(self.hparams.batch_size, temperature)

    def forward(self, batch, projection: bool = True):
        data1, data2 = batch

        h1, h2 = self.module(data1), self.module(data2)

        if not projection:
            return h1, h2

        return self.projection(h1), self.projection(h2)

    def training_step(self, batch, batch_idx):
        h1, h2 = self(batch)
        # loss = self.contrastive_loss(h1, h2)

        h1, h2 = self.sigmoid(self.classifier(h1)), self.sigmoid(self.classifier(h2))
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
        # dataset = PairsDataset(
        #     self.adata,
        #     self.x,
        #     self.embedding,
        #     self.hparams.n_hops,
        #     n_intermediate=self.hparams.n_intermediate,
        # )
        dataset = ShuffledDataset(
            self.adata,
            self.x,
            self.embedding,
            self.hparams.n_hops,
            n_intermediate=self.hparams.n_intermediate,
        )
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )

    # @torch.no_grad()
    # def embeddings(self):
    #     if importlib.util.find_spec("ipywidgets") is not None:
    #         from tqdm.autonotebook import tqdm
    #     else:
    #         from tqdm import tqdm

    #     dataset = StandardDataset(self.adata, self.x, self.embedding, self.hparams.n_hops)
    #     loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)
    #     return torch.concatenate(
    #         [self.module(batch) for batch in tqdm(loader, desc="DataLoader")]
    #     )

    # def baseline_embeddings(self):
    #     sampler = HopSampler(self.adata, self.x)

    #     import numpy as np

    #     return np.stack(
    #         [
    #             self.x_numpy[
    #                 sampler.hop_travel(i, self.hparams.n_hops, as_pyg=False)
    #             ].mean(0)
    #             for i in range(self.adata.n_obs)
    #         ]
    #     )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
