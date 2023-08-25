import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.models import GAT


class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """

    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = torch.diag_embed(torch.full((batch_size * 2,), -torch.inf))

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
        )

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        cos_sim = self.calc_similarity_batch(z_i, z_j) / self.temperature

        sim_ij = torch.diag(cos_sim, self.batch_size)
        sim_ji = torch.diag(cos_sim, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        all_losses = -positives + torch.logsumexp(cos_sim + self.mask, dim=-1)
        return all_losses.mean()


class GraphEncoder(pl.LightningModule):
    def __init__(self, num_features, hidden_channels, num_layers, out_channels) -> None:
        super().__init__()
        self.gat = GAT(
            num_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            v2=True,
        )

        self.seq = nn.Sequential(nn.Linear(out_channels, 1), nn.Sigmoid())
        self.attention_aggregation = AttentionalAggregation(self.seq)

    def forward(self, data: Data):
        out = self.gat(x=data.x, edge_index=data.edge_index)
        h = self.attention_aggregation(out, ptr=data.ptr)
        return h


class GraphCL(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 64,
        num_layers: int = 5,
        out_channels: int = 64,
        batch_size: int = 32,
        lr: float = 1e-3,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.module = GraphEncoder(
            num_features, hidden_channels, num_layers, out_channels
        )
        self.projection = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        self.batch_size = batch_size
        self.lr = lr
        self.contrastive_loss = ContrastiveLoss(self.batch_size, temperature)

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
            batch_size=self.batch_size,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
