from __future__ import annotations

import lightning as L
import torch
from torch import Tensor, nn
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.utils import scatter, softmax

from .. import settings, utils
from .._constants import Nums


class AttentionAggregation(Aggregation, L.LightningModule):
    """Aggregate the node embeddings using attention."""

    @utils.format_docs
    def __init__(self, output_size: int):
        """

        Args:
            {output_size}
        """
        super().__init__()
        self.gate_nn = nn.Linear(output_size, 1)
        self.nn = nn.Linear(output_size, output_size)

        self._entropies = torch.tensor([], dtype=torch.float32)

    def forward(
        self,
        x: Tensor,
        index: Tensor | None = None,
        ptr: None = None,
        dim_size: None = None,
        dim: int = -2,
    ) -> Tensor:
        """Performs attention aggragation.

        Args:
            x: The nodes embeddings representing `B` total graphs.
            index: The Pytorch Geometric index used to know to which graph each node belongs.

        Returns:
            A tensor of shape `(B, O)` of graph embeddings.
        """
        gate = self.gate_nn(x)
        x = self.nn(x)

        gate = softmax(gate, index, dim=dim)

        if settings.store_attention_entropy:
            attention_entropy = scatter(-gate * (gate + Nums.EPS).log2(), index=index)[:, 0]
            self._entropies = torch.cat([self._entropies, attention_entropy])

        return self.reduce(gate * x, index, dim=dim)

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gate_nn={self.gate_nn}, nn={self.nn})"
