import importlib

import numpy as np
import pytorch_lightning as pl
import torch
from anndata import AnnData
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from .data import LocalAugmentationDataset
from .module import GenesEmbedding, GraphEncoder, SwavHead


class GraphCL(pl.LightningModule):
    def __init__(
        self,
        adata: AnnData,
        swav: bool,
        batch_key: str = None,
        # obsm_key="X_pca",
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
        self.save_hyperparameters(ignore=["adata"])

        self.adata = adata
        self.swav = swav
        self.x_numpy = self.adata.X  # log1p expressions
        # TODO: keep? how with multi adata?
        self.x_numpy = (self.x_numpy - self.x_numpy.mean(0)) / self.x_numpy.std(0)
        self.x = torch.tensor(self.x_numpy)

        self.embedding = GenesEmbedding(adata.var_names, embedding_size)

        # PCA init embeddings (valid only if x centered)
        pca = PCA(n_components=embedding_size)
        pca.fit(self.x_numpy)
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

        if self.swav:
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
        dataset = LocalAugmentationDataset(
            self.adata,
            self.x,
            self.embedding,
            delta_th=0.5,
            n_hops=self.hparams.n_hops,
            n_intermediate=self.hparams.n_intermediate,
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)

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

    def on_train_epoch_start(self):
        self.swav_head.prototypes.requires_grad = self.current_epoch > 0

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
    def swav_clusters(self) -> np.ndarray:
        preds = []

        loader = self.test_dataloader()

        for h1, _, _ in loader:
            np_, _ = self.module(h1)
            out1 = F.normalize(np_, dim=1, p=2)
            scores1 = out1 @ self.swav_head.prototypes
            pred = scores1.argmax(1)

            preds.append(pred)

        preds = torch.cat(preds)

        res = np.full(self.adata.n_obs, "nan")
        res[loader.dataset.valid_indices] = preds.numpy(force=True).astype(str)

        return res

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
