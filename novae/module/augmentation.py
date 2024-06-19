import lightning as L
import torch
from torch.distributions import Exponential
from torch_geometric.data import Data


class GraphAugmentation(L.LightningModule):
    def __init__(
        self,
        panel_subset_size: float,
        background_noise_lambda: float,
        sensitivity_noise_std: float,
    ):
        super().__init__()
        self.panel_subset_size = panel_subset_size
        self.background_noise_lambda = background_noise_lambda
        self.sensitivity_noise_std = sensitivity_noise_std

        self.background_noise_distribution = Exponential(torch.tensor(float(background_noise_lambda)))

    def noise(self, data: Data):
        sample_shape = (data.batch_size, data.x.shape[1])

        additions = self.background_noise_distribution.sample(sample_shape=sample_shape).to(self.device)
        gaussian_noise = torch.randn(sample_shape, device=self.device)
        factors = (1 + gaussian_noise * self.sensitivity_noise_std).clip(0, 2)

        for i in range(data.batch_size):
            start, stop = data.ptr[i], data.ptr[i + 1]
            data.x[start:stop] = data.x[start:stop] * factors[i] + additions[i]

    def panel_subset(self, data: Data):
        n_total = len(data.genes_indices[0])
        n_subset = int(n_total * self.panel_subset_size)

        gene_subset_indices = torch.randperm(n_total)[:n_subset]

        data.x = data.x[:, gene_subset_indices]
        data.genes_indices = data.genes_indices[:, gene_subset_indices]

    def forward(self, data: Data) -> Data:
        self.noise(data)
        self.panel_subset(data)
        return data
