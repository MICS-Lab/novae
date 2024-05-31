from __future__ import annotations

import lightning as L
from torch_geometric.data import Data
from torch_geometric.nn.models import GAT

from .aggregation import NodeAttentionAggregation


class GraphEncoder(L.LightningModule):
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

        self.node_aggregation = NodeAttentionAggregation(out_channels)

    def forward(self, data: Data):
        return self.gnn(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)

    def node_x(self, data: Data):
        out = self(data)
        return self.node_aggregation(out, index=data.batch)
