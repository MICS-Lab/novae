from __future__ import annotations

import lightning as L
from torch_geometric.data import Data
from torch_geometric.nn.models import GAT

from .aggregation import NodeAttentionAggregation


class GraphEncoder(L.LightningModule):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.gnn = GAT(
            embedding_size,
            hidden_channels=hidden_size,
            num_layers=num_layers,
            out_channels=output_size,
            edge_dim=1,
            v2=True,
            heads=heads,
            act="ELU",
        )

        self.node_aggregation = NodeAttentionAggregation(output_size)

    def forward(self, data: Data):
        out = self.gnn(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        return self.node_aggregation(out, index=data.batch)
