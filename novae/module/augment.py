import lightning as L
import torch
from torch.distributions import Exponential
from torch_geometric.data import Data

from .. import utils


class GraphAugmentation(L.LightningModule):
    """Perform graph augmentation for Novae. It adds noise to the data and keeps a subset of the genes."""

    @utils.format_docs
    def __init__(
        self,
        panel_subset_size: float,
        background_noise_lambda: float,
        sensitivity_noise_std: float,
    ):
        """

        Args:
            {panel_subset_size}
            {background_noise_lambda}
            {sensitivity_noise_std}
        """
        super().__init__()
        self.panel_subset_size = panel_subset_size
        self.background_noise_lambda = background_noise_lambda
        self.sensitivity_noise_std = sensitivity_noise_std

        self.background_noise_distribution = Exponential(torch.tensor(float(background_noise_lambda)))

    @utils.format_docs
    def noise(self, data: Data):
        """Add noise (inplace) to the data as detailed in the article.

        Args:
            {data}
        """
        sample_shape = (data.batch_size, data.x.shape[1])

        additions = self.background_noise_distribution.sample(sample_shape=sample_shape).to(self.device)
        gaussian_noise = torch.randn(sample_shape, device=self.device)
        factors = (1 + gaussian_noise * self.sensitivity_noise_std).clip(0, 2)

        for i in range(data.batch_size):
            start, stop = data.ptr[i], data.ptr[i + 1]
            data.x[start:stop] = data.x[start:stop] * factors[i] + additions[i]

    @utils.format_docs
    def panel_subset(self, data: Data):
        """
        Keep a ratio of `panel_subset_size` of the input genes (inplace operation).

        Args:
            {data}
        """
        n_total = len(data.genes_indices[0])
        n_subset = int(n_total * self.panel_subset_size)

        gene_subset_indices = torch.randperm(n_total)[:n_subset]

        data.x = data.x[:, gene_subset_indices]
        data.genes_indices = data.genes_indices[:, gene_subset_indices]

    @utils.format_docs
    def forward(self, data: Data) -> Data:
        """Perform data augmentation (`noise` and `panel_subset`).

        Args:
            {data}

        Returns:
            The augmented `Data` object
        """
        self.panel_subset(data)
        self.noise(data)
        return data
