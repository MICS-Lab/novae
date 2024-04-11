from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import to_scipy_sparse_matrix


class NodeAttentionAggregation(L.LightningModule):
    def __init__(self, out_channels: int):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(out_channels, 1), nn.Sigmoid())
        self.attention_aggregation = AttentionalAggregation(self.seq)

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        return self.attention_aggregation(x, index=index)


class EdgeAttentionAggregation(L.LightningModule):
    """
    A PyTorch Lightning module for aggregating edge attention scores in graph neural networks.

    This module computes edge attention scores using a defined edge scoring mechanism and
    provides options for aggregating these scores. The aggregation can be performed either
    by directly returning the edge scores per batch in the form of scipy sparse matrices
    or by applying a global mean pooling operation over the computed scores.

    Parameters:
    - in_channels (int): The number of input channels (features) of each node in the graph.
    - out_channels (int): The number of output channels (features) for each edge score computed.
    - heads (int, optional): The number of attention heads used in the edge scoring mechanism.
      Defaults to 1.

    Attributes:
    - edge_scorer (EdgeScorer): An instance of the EdgeScorer class, initialized with the provided
      `in_channels`, `out_channels`, and `heads` parameters. This scorer is responsible for
      computing the attention scores for each edge in the graph.

    Methods:
    - forward(x, edge_index, batch, return_weights=False): Computes the forward pass of the module.
      Based on the `return_weights` flag, it either returns edge scores per batch as scipy sparse
      matrices or applies global mean pooling on the scores.
    - _get_edge_scores_per_batch(scores, edge_index, batch): A helper method that computes the edge
      scores per batch and returns them as a list of scipy sparse matrices.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, **kwargs):
        super().__init__()
        self.edge_scorer = EdgeScorer(in_channels, out_channels, heads, **kwargs)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, return_weights: bool = False) -> torch.Tensor:
        """
        Performs the forward pass of the EdgeAttentionAggregation module.

        Parameters:
        - x (Tensor): The node features tensor of shape [num_nodes, in_channels].
        - edge_index (Tensor): The edge indices tensor of shape [2, num_edges].
        - batch (Tensor): The batch tensor, indicating the graph to which each node belongs.
        - return_weights (bool, optional): If True, returns the computed edge scores per batch
          as scipy sparse matrices. If False, returns a tensor aggregated by global mean pooling.
          Defaults to False.

        Returns:
        - torch.Tensor | list: Depending on `return_weights`, either returns a tensor of aggregated
          scores or a list of scipy sparse matrices representing the edge scores per batch.
        """
        scores = self.edge_scorer(x=x, edge_index=edge_index)
        if return_weights:
            return self._get_edge_scores_per_batch(scores, edge_index, batch[edge_index[0]])
        else:
            return global_mean_pool(x=scores, batch=batch[edge_index[0]])

    def _get_edge_scores_per_batch(self, scores: Tensor, edge_index: Tensor, batch: Tensor):
        """
        Computes edge scores per batch and returns them as scipy sparse matrices.

        This helper method isolates the edges and their corresponding scores for each unique
        batch in the graph data and constructs a scipy sparse matrix for each batch.

        Parameters:
        - scores (Tensor): The tensor of computed edge scores.
        - edge_index (Tensor): The tensor of edge indices.
        - batch (Tensor): The tensor indicating the batch ID for each edge.

        Returns:
        - list: A list of scipy sparse matrices, each representing the edge scores for a graph
          in the batch.
        """
        batches = torch.unique(batch)
        edge_scores_per_batch = []
        for batch_idx in batches:
            graph_per_batch = to_scipy_sparse_matrix(
                edge_index=edge_index[:, batch == batch_idx] - torch.min(edge_index[:, batch == batch_idx]).item(),
                edge_attr=scores[batch == batch_idx],
            )
            edge_scores_per_batch.append(graph_per_batch)
        return edge_scores_per_batch


class EdgeScorer(MessagePassing, L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_dim: int | None = None,
        fill_value: float | Tensor | str = "mean",
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias, weight_initializer="glorot")
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias, weight_initializer="glorot")

        self.att = Parameter(torch.empty(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(  # noqa: F811
        self,
        x: Tensor,
        edge_index: Adj,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None

        assert x.dim() == 2
        x_l = self.lin_l(x).view(-1, H, C)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        alpha = self.edge_updater(edge_index, x=(x_l, x_r))

        alpha = torch.sigmoid(alpha.mean(1))

        return alpha

    def edge_update(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        x = x_i + x_j

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, " f"{self.out_channels}, heads={self.heads})"
