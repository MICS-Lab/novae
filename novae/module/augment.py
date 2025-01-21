import lightning as L
import torch
from torch import Tensor
from torch.distributions import Exponential
from torch_geometric.data import Data


class GraphAugmentation(L.LightningModule):
    """Perform graph augmentation for Novae. It adds noise to the data and keeps a subset of the genes."""

    def __init__(
        self,
        panel_subset_size: float,
        background_noise_lambda: float,
        sensitivity_noise_std: float,
    ):
        """

        Args:
            panel_subset_size: Ratio of genes kept from the panel during augmentation.
            background_noise_lambda: Parameter of the exponential distribution for the noise augmentation.
            sensitivity_noise_std: Standard deviation for the multiplicative for for the noise augmentation.
        """
        super().__init__()
        self.panel_subset_size = panel_subset_size
        self.background_noise_lambda = background_noise_lambda
        self.sensitivity_noise_std = sensitivity_noise_std

        self.background_noise_distribution = Exponential(torch.tensor(float(background_noise_lambda)))

    def noise(self, data: Data):
        """Add noise (inplace) to the data as detailed in the article.

        Args:
            data: A Pytorch Geometric `Data` object representing a graph.
        """
        sample_shape = (data.x.shape[1],)

        additions = self.background_noise_distribution.sample(sample_shape=sample_shape).to(self.device)
        gaussian_noise = torch.randn(sample_shape, device=self.device)
        factors = (1 + gaussian_noise * self.sensitivity_noise_std).clip(0, 2)

        data.x = data.x * factors + additions

    def panel_subset(self, data: Data, genes_indices: Tensor) -> Tensor:
        """
        Keep a ratio of `panel_subset_size` of the input genes (inplace operation).

        Args:
            data: A Pytorch Geometric `Data` object representing a graph.

        Returns:
            The new gene indices.
        """
        n_total = len(genes_indices)
        n_subset = int(n_total * self.panel_subset_size)

        gene_subset_indices = torch.randperm(n_total)[:n_subset]

        data.x = data.x[:, gene_subset_indices]

        return gene_subset_indices

    def forward(self, data: Data, genes_indices: Tensor) -> tuple[Data, Tensor]:
        """Perform data augmentation (`noise` and `panel_subset`).

        Args:
            data: A Pytorch Geometric `Data` object representing a batch of `B` graphs.
            genes_indices: A `Tensor` of gene indices to use for the embedding.

        Returns:
            The augmented `Data` object and the new gene indices.
        """
        gene_subset_indices = self.panel_subset(data, genes_indices)
        self.noise(data)
        return data, gene_subset_indices
