import lightning as L
import torch
from torch.distributions import Exponential

from .embedding import GenesEmbedding


class GraphAugmentation(L.LightningModule):
    def __init__(
        self,
        genes_embedding: GenesEmbedding,
        panel_dropout: float = 0.2,
        gene_expression_dropout: float = 0.1,
        background_noise_lambda: float = 5.0,
        sensitivity_noise_std: float = 0.05,
    ):
        super().__init__()
        self.genes_embedding = genes_embedding
        self.save_hyperparameters(ignore="genes_embedding")

        self.background_noise_distribution = Exponential(torch.tensor(background_noise_lambda))

    def forward(self, x: torch.Tensor, var_names: list[str], ignore: bool = False) -> torch.Tensor:
        genes_indices = self.genes_embedding.genes_to_indices(var_names)

        if ignore:
            return self.genes_embedding(x, genes_indices)

        # noise background + sensitivity
        addition = self.background_noise_distribution.sample(sample_shape=(x.shape[1],)).to(
            self.device
        )
        factor = (
            1 + torch.randn(x.shape[1], device=self.device) * self.hparams.sensitivity_noise_std
        ).clip(0, 2)
        x = x * factor + addition

        # gene expression dropout (= low quality gene)
        # indices = torch.randperm(x.shape[1])[: int(x.shape[1] * self.gene_expression_dropout)]
        # x[:, indices] = 0

        # gene subset (= panel change)
        n_vars = len(genes_indices)
        gene_subset_indices = torch.randperm(n_vars)[
            : int(n_vars * (1 - self.hparams.panel_dropout))
        ]

        x = self.genes_embedding(x[:, gene_subset_indices], genes_indices[gene_subset_indices])

        return x
