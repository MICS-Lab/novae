from __future__ import annotations

import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GAT

from .pooling import EdgeAttentionPooling, NodeAttentionPooling


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

        self.node_pooling = NodeAttentionPooling(out_channels)
        self.edge_pooling = EdgeAttentionPooling(out_channels, out_channels, heads=heads)

    def forward(self, data: Data):
        return self.gnn(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)

    def node_x(self, data: Data):
        out = self(data)
        return self.node_pooling(out, ptr=data.ptr)

    def edge_x(self, data: Data):
        out = self(data)
        x = self.edge_pooling(x=out, edge_index=data.edge_index)
        x = global_mean_pool(x=x, batch=data.batch[data.edge_index[0]])
        return x
