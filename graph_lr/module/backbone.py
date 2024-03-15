from __future__ import annotations

import pytorch_lightning as pl
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.models import GAT

from .head import EdgeScorer


class GraphEncoder(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.gnn = GAT(
            num_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            edge_dim=1,
            v2=True,
            heads=heads,
            act="ELU",
        )

        # Node pooling
        self.seq = nn.Sequential(nn.Linear(out_channels, 1), nn.Sigmoid())
        self.attention_aggregation = AttentionalAggregation(self.seq)

        # Edge pooling
        self.edge_scorer = EdgeScorer(out_channels, out_channels, heads=heads)

    def forward(self, data: Data):
        out = self.gnn(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)

        node_pooling = self.attention_aggregation(out, ptr=data.ptr)

        scores = self.edge_scorer(x=out, edge_index=data.edge_index)
        edge_pooling = global_mean_pool(x=scores, batch=data.batch[data.edge_index[0]])

        return node_pooling, edge_pooling
