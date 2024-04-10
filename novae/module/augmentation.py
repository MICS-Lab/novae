import lightning as L
import torch
from torch.distributions import Exponential


class GraphAugmentation(L.LightningModule):
    def __init__(
        self,
        panel_dropout: float = 0.2,
        gene_expression_dropout: float = 0.1,
        background_noise_lambda: float = 5.0,
        sensitivity_noise_std: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.background_noise_distribution = Exponential(torch.tensor(background_noise_lambda))

    def noise(self, x: torch.Tensor) -> torch.Tensor:
        addition = self.background_noise_distribution.sample(sample_shape=(x.shape[1],))

        gaussian_noise = torch.randn(x.shape[1], device=self.device)
        factor = (1 + gaussian_noise * self.hparams.sensitivity_noise_std).clip(0, 2)

        return x * factor + addition.to(self.device)

    def panel_subset(self, x: torch.Tensor, genes_indices: torch.Tensor) -> torch.Tensor:
        n_total = len(genes_indices)
        n_subset = int(n_total * (1 - self.hparams.panel_dropout))

        gene_subset_indices = torch.randperm(n_total)[:n_subset]

        return x[:, gene_subset_indices], genes_indices[gene_subset_indices]

    def forward(self, x: torch.Tensor, genes_indices: torch.Tensor) -> torch.Tensor:
        x = self.noise(x)
        x, genes_indices = self.panel_subset(x, genes_indices)

        return x, genes_indices
