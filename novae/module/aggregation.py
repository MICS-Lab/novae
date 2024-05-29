from __future__ import annotations

import lightning as L
from torch import Tensor, nn
from torch_geometric.nn.aggr import AttentionalAggregation


class NodeAttentionAggregation(L.LightningModule):
    def __init__(self, out_channels: int):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(out_channels, 1), nn.Sigmoid())
        self.attention_aggregation = AttentionalAggregation(self.seq)

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        return self.attention_aggregation(x, index=index)
