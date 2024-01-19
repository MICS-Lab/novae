import typing
from typing import Any, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.data import Data

# from torch_geometric.nn import aggr
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.models import GAT, GCN, GraphSAGE
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.typing import Adj, NoneType, OptTensor, PairTensor, torch_sparse
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value


class GATv2ConvFinal(MessagePassing):
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

        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, x=(x_l, x_r))

        alpha = torch.sigmoid(alpha.mean(1))

        return alpha

        # # propagate_type: (x: PairTensor, alpha: Tensor)
        # out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        # if self.concat:
        #     out = out.view(-1, self.heads * self.out_channels)
        # else:
        #     out = out.mean(dim=1)

        # if self.bias is not None:
        #     out = out + self.bias

        # if isinstance(return_attention_weights, bool):
        #     if isinstance(edge_index, Tensor):
        #         if is_torch_sparse_tensor(edge_index):
        #             # TODO TorchScript requires to return a tuple
        #             adj = set_sparse_value(edge_index, alpha)
        #             return out, (adj, alpha)
        #         else:
        #             return out, (edge_index, alpha)
        # else:
        #     return out

    def edge_update(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        x = x_i + x_j

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        # alpha = softmax(alpha, index, ptr, dim_size)
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


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


# class Embedding(pl.LightningModule):
#     def __init__(
#         self, voc_size: int, embedding_size: int, drop_ratio: float = 0.1
#     ) -> None:
#         super().__init__()
#         self.voc_size = voc_size
#         self.embeddings = nn.Embedding(voc_size, embedding_size)
#         self.softmax = nn.Softmax(dim=1)
#         self.dropped_voc_size = int(self.voc_size * (1 - drop_ratio))

#     def forward(self, x: torch.Tensor, drop_nodes: bool = False) -> torch.Tensor:
#         if not drop_nodes:
#             emb_weights = self.softmax(self.embeddings.weight)
#             return x @ emb_weights

#         indices = torch.randperm(self.voc_size)[: self.dropped_voc_size]
#         emb_weights = self.softmax(self.embeddings(indices))
#         return x[:, indices] @ emb_weights


class ContrastiveLoss(pl.LightningModule):
    """Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper"""

    def __init__(self, batch_size: int, temperature: float):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = torch.diag_embed(torch.full((batch_size * 2,), -torch.inf))

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
        )

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs z_i, z_j in the SimCLR paper
        """
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        cos_sim = self.calc_similarity_batch(z_i, z_j) / self.temperature

        sim_ij = torch.diag(cos_sim, self.batch_size)
        sim_ji = torch.diag(cos_sim, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        return (-positives + torch.logsumexp(cos_sim + self.mask, dim=-1)).mean()


class Transformer(BasicGNN):
    supports_edge_weight: bool = False
    supports_edge_attr: bool = False

    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return TransformerConv(in_channels, out_channels, **kwargs)


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

        # self.gnn = Transformer(
        #     num_features,
        #     hidden_channels=hidden_channels,
        #     num_layers=num_layers,
        #     out_channels=out_channels,
        #     heads=heads,
        #     concat=False,
        # )

        self.seq = nn.Sequential(nn.Linear(out_channels, 1), nn.Sigmoid())
        self.attention_aggregation = AttentionalAggregation(self.seq)

        # self.edge_scorer = nn.Sequential(
        #     nn.Linear(2 * out_channels, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid(),
        # )

        self.edge_scorer = GATv2ConvFinal(out_channels, out_channels, heads=heads)

    def forward(self, data: Data):
        out = self.gnn(x=data.x, edge_index=data.edge_index)
        # h = self.attention_aggregation(out, ptr=data.ptr)
        # return h

        # edge_embeddings = torch.cat(
        #     (out[data.edge_index[0]], out[data.edge_index[1]]), dim=1
        # )
        scores = self.edge_scorer(x=out, edge_index=data.edge_index)
        return global_mean_pool(x=scores, batch=data.batch[data.edge_index[0]])
