import lightning as L
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn.models import GAT

from . import AttentionAggregation


class GraphEncoder(L.LightningModule):
    """Graph encoder of Novae. It uses a graph attention network."""

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        heads: int,
        histo_embedding_size: int,
    ) -> None:
        """
        Args:
            embedding_size: Size of the embeddings of the genes (`E` in the article).
            hidden_size: The size of the hidden layers in the GAT.
            num_layers: The number of layers in the GAT.
            output_size: Size of the representations, i.e. the encoder outputs (`O` in the article).
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

        self.mlp_fusion = nn.Sequential(
            nn.Linear(histo_embedding_size + output_size, histo_embedding_size + output_size),
            nn.ReLU(),
            nn.Linear(histo_embedding_size + output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, data: Batch) -> Tensor:
        """Encode the input data.

        Args:
            data: A Pytorch Geometric `Data` object representing a batch of `B` graphs.

        Returns:
            A tensor of shape `(B, O)` containing the encoded graphs.
        """
        out = self.gnn(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        out = self.node_aggregation(out, index=data.batch)

        if hasattr(data, "histo_embeddings"):
            out = self.mlp_fusion(torch.cat([out, data.histo_embeddings], dim=-1))

        return out
