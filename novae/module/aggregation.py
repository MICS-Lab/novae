from __future__ import annotations

import lightning as L
from torch import Tensor, nn
from torch_geometric.nn.aggr import AttentionalAggregation


class NodeAttentionAggregation(L.LightningModule):
    def __init__(self, out_channels: int):
        super().__init__()
        self.gate_nn = nn.Linear(out_channels, 1)
        self.nn = nn.Linear(out_channels, out_channels)

        self.attention_aggregation = AttentionalAggregation(gate_nn=self.gate_nn, nn=self.nn)

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        return self.attention_aggregation(x, index=index)
