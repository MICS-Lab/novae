from __future__ import annotations

import lightning as L
from torch import Tensor, nn
from torch_geometric.nn.aggr import AttentionalAggregation


class AttentionAggregation(L.LightningModule):
    """Aggregate the node embeddings using attention."""

    def __init__(self, output_size: int):
        """

        Args:
            output_size: Size of the nodes after the encoder.
        """
        super().__init__()
        self.gate_nn = nn.Linear(output_size, 1)
        self.nn = nn.Linear(output_size, output_size)

        self.attention_aggregation = AttentionalAggregation(gate_nn=self.gate_nn, nn=self.nn)

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        """Performs attention aggragation.

        Args:
            x: The nodes embeddings representing `B` total graphs.
            index: The Pytorch Geometric index used to know to which graph each node belongs.

        Returns:
            A tensor of shape `(B, output_size)` of graph embeddings.
        """
        return self.attention_aggregation(x, index=index)