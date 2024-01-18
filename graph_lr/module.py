from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

# from torch_geometric.nn import aggr
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.models import GAT, GCN, GraphSAGE
from torch_geometric.nn.models.basic_gnn import BasicGNN


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

        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * out_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: Data):
        out = self.gnn(x=data.x, edge_index=data.edge_index)
        # h = self.attention_aggregation(out, ptr=data.ptr)
        # return h

        edge_embeddings = torch.cat(
            (out[data.edge_index[0]], out[data.edge_index[1]]), dim=1
        )
        scores = self.edge_scorer(edge_embeddings)
        return global_mean_pool(x=scores, batch=data.batch[data.edge_index[0]])
