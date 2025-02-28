import lightning as L
from torch import Tensor, nn
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax


class AttentionAggregation(Aggregation, L.LightningModule):
    """Aggregate the node embeddings using attention."""

    def __init__(self, output_size: int):
        """

        Args:
            output_size: Size of the representations, i.e. the encoder outputs (`O` in the article).
        """
        super().__init__()
        self.attention_aggregation = ProjectionLayers(output_size)  # for backward compatibility when loading models

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
        gate = self.attention_aggregation.gate_nn(x)
        x = self.attention_aggregation.nn(x)

        gate = softmax(gate, index, dim=dim)

        return self.reduce(gate * x, index, dim=dim)

    def reset_parameters(self):
        reset(self.attention_aggregation.gate_nn)
        reset(self.attention_aggregation.nn)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gate_nn={self.attention_aggregation.gate_nn}, nn={self.attention_aggregation.nn})"


class ProjectionLayers(L.LightningModule):
    """
    Small class for backward compatibility when loading models
    Contains the projection layers used for the attention aggregation
    """

    def __init__(self, output_size):
        super().__init__()
        self.gate_nn = nn.Linear(output_size, 1)
        self.nn = nn.Linear(output_size, output_size)
