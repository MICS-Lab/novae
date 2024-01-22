from typing import Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.models import GAT
from torch_geometric.typing import Adj, OptTensor


class GenesEmbedding(pl.LightningModule):
    def __init__(self, gene_names: list[str], embedding_size: int) -> None:
        super().__init__()
        self.voc_size = len(gene_names)
        self.gene_to_index = {gene: i for i, gene in enumerate(gene_names)}

        self.embedding = nn.Embedding(self.voc_size, embedding_size)
        self.softmax = nn.Softmax(dim=0)

    def genes_to_indices(self, gene_names: list[str]) -> torch.Tensor:
        return torch.tensor(
            [self.gene_to_index[gene] for gene in gene_names], dtype=torch.long
        )

    def forward(self, x: torch.Tensor, genes_indices: torch.Tensor) -> torch.Tensor:
        genes_embeddings = self.embedding(genes_indices)
        genes_embeddings = self.softmax(genes_embeddings)

        return x @ genes_embeddings


class SwavHead(pl.LightningModule):
    def __init__(self, out_channels: int, num_prototypes: int):
        self.out_channels = out_channels
        self.num_prototypes = num_prototypes

        self.prototypes = nn.Linear(self.out_channels, self.num_prototypes, bias=False)

    @torch.no_grad()
    def sinkhorn(out, epsilon: float = 0.05, sinkhorn_iterations: int = 3):
        """Q is K-by-B for consistency with notations from the paper (out: B*K)"""
        Q = torch.exp(out / epsilon).t()
        Q /= torch.sum(Q)

        B = Q.shape[1]
        K = Q.shape[0]

        for _ in range(sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()


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
            v2=True,
            heads=heads,
        )

        # Node pooling
        self.seq = nn.Sequential(nn.Linear(out_channels, 1), nn.Sigmoid())
        self.attention_aggregation = AttentionalAggregation(self.seq)

        # Edge pooling
        self.edge_scorer = EdgeScorer(out_channels, out_channels, heads=heads)

    def forward(self, data: Data):
        out = self.gnn(x=data.x, edge_index=data.edge_index)

        node_pooling = self.attention_aggregation(out, ptr=data.ptr)

        scores = self.edge_scorer(x=out, edge_index=data.edge_index)
        edge_pooling = global_mean_pool(x=scores, batch=data.batch[data.edge_index[0]])

        return node_pooling, edge_pooling


class EdgeScorer(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
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

        self.lin_l = Linear(
            in_channels, heads * out_channels, bias=bias, weight_initializer="glorot"
        )
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(
                in_channels, heads * out_channels, bias=bias, weight_initializer="glorot"
            )

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
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]],]:
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
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )
