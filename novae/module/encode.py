from __future__ import annotations

import lightning as L
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.models import GAT

from .. import utils
from . import AttentionAggregation


class GraphEncoder(L.LightningModule):
    """Graph encoder of Novae. It uses a graph attention network."""

    @utils.format_docs
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        heads: int,
    ) -> None:
        """
        Args:
            {embedding_size}
            hidden_size: The size of the hidden layers in the GAT.
            num_layers: The number of layers in the GAT.
            {output_size}
            heads: The number of attention heads in the GAT.
        """
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

        self.node_aggregation = AttentionAggregation(output_size)

    @utils.format_docs
    def forward(self, data: Data) -> Tensor:
        """Encode the input data.

        Args:
            {data}

        Returns:
            A tensor of shape `(B, O)` containing the encoded graphs.
        """
        out = self.gnn(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        return self.node_aggregation(out, index=data.batch)
