import logging

import lightning as L
import torch
from torch.distributions import Exponential, Normal
from torch_geometric.data import Batch

log = logging.getLogger(__name__)


class GraphAugmentation(L.LightningModule):
    """Perform graph augmentation for Novae. It adds noise to the data and keeps a subset of the genes."""

    def __init__(
        self,
        panel_subset_size: float,
        background_noise_lambda: float,
        sensitivity_noise_std: float,
        dropout_rate: float,
        use_repr: bool,
    ):
        """

        Args:
            panel_subset_size: Ratio of genes kept from the panel during augmentation.
            background_noise_lambda: Parameter of the exponential distribution for the noise augmentation. If `use_repr` use a normal distribution with std = lambda instead.
            sensitivity_noise_std: Standard deviation for the multiplicative for for the noise augmentation.
            dropout_rate: Dropout rate for the genes during augmentation.
            use_repr: Whether to use representations instead of raw data for augmentation.
        """
        super().__init__()
        self.panel_subset_size = panel_subset_size
        self.background_noise_lambda = background_noise_lambda
        self.sensitivity_noise_std = sensitivity_noise_std
        self.dropout_rate = dropout_rate
        self.use_repr = use_repr

        if self.use_repr:
            self.background_noise_distribution = Normal(0, float(background_noise_lambda))
        else:
            self.background_noise_distribution = Exponential(torch.tensor(float(background_noise_lambda * 0.1)))
            if self.panel_subset_size != 0:
                log.warning("`panel_subset_size != 0` but it will not be used since `embedding_name` is used")

    def noise(self, data: Batch):
        """Add noise (inplace) to the data as detailed in the article.

        Args:
            data: A Pytorch Geometric `Data` object representing a batch of `B` graphs.
        """
        sample_shape = (data.batch_size, data.x.shape[1])

        additions = self.background_noise_distribution.sample(sample_shape=sample_shape).to(self.device)
        gaussian_noise = torch.randn(sample_shape, device=self.device)
        factors = (1 + gaussian_noise * self.sensitivity_noise_std).clip(0, 2)

        for i in range(data.batch_size):
            start, stop = data.ptr[i], data.ptr[i + 1]
            data.x[start:stop] = data.x[start:stop] * factors[i] + additions[i]

    def dropout(self, data: Batch):
        """**Deprecated**. Set to 0 the expression of some genes (inplace).

        Args:
            data: A Pytorch Geometric `Data` object representing a batch of `B` graphs.
        """
        if self.dropout_rate == 0:
            return

        mask = torch.rand(data.x.shape[1], device=self.device) < self.dropout_rate
        data.x[:, mask] = 0

    def panel_subset(self, data: Batch):
        """
        Keep a ratio of `panel_subset_size` of the input genes (inplace operation).
        Contrary to the dropout, it doesn't set the expression to 0, but removes the genes from the panel.

        Args:
            data: A Pytorch Geometric `Data` object representing a batch of `B` graphs.
        """
        n_total = len(data.genes_indices[0])
        n_subset = int(n_total * self.panel_subset_size)

        gene_subset_indices = torch.randperm(n_total)[:n_subset]

        data.x = data.x[:, gene_subset_indices]
        data.genes_indices = data.genes_indices[:, gene_subset_indices]

    def forward(self, data: Batch) -> Batch:
        """Perform data augmentation (`noise` and `panel_subset`).

        Args:
            data: A Pytorch Geometric `Data` object representing a batch of `B` graphs.

        Returns:
            The augmented `Data` object
        """
        if not self.use_repr:
            self.panel_subset(data)

        self.noise(data)

        return data
