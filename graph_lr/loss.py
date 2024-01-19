import pytorch_lightning as pl
import torch
import torch.nn.functional as F


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
