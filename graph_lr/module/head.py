from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class SwavHead(pl.LightningModule):
    def __init__(
        self,
        out_channels: int,
        num_prototypes: int,
        temperature: float = 0.1,
        queue_size: int = 1000,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.queue_size = queue_size

        if self.queue_size is not None:
            self.register_buffer(
                "queue", torch.zeros((self.queue_size, out_channels), dtype=torch.float32)
            )
        else:
            self.queue = None

        self.prototypes = nn.Parameter(torch.empty((self.out_channels, self.num_prototypes)))
        self.prototypes = nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))
        self.normalize_prototypes()

    @property
    def use_queue(self) -> bool:
        return self.queue is not None and self.current_epoch >= 5

    def normalize_prototypes(self):
        self.prototypes.data = F.normalize(self.prototypes.data, dim=1, p=2)

    def forward(self, out1, out2):
        self.normalize_prototypes()

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        scores1 = out1 @ self.prototypes
        scores2 = out2 @ self.prototypes

        if self.use_queue:
            scores_queued1 = torch.cat([scores1.detach(), self.queue @ self.prototypes])
            scores_queued2 = torch.cat([scores2.detach(), self.queue @ self.prototypes])

            q1 = self.sinkhorn(scores_queued1)[: len(scores1)]
            q2 = self.sinkhorn(scores_queued2)[: len(scores2)]
        else:
            q1 = self.sinkhorn(scores1)
            q2 = self.sinkhorn(scores2)

        if self.queue is not None:
            n = len(out1) + len(out2)
            self.queue[n:] = self.queue[:-n].clone()
            self.queue[:n] = torch.cat([out1.detach(), out2.detach()], axis=0)

        return -0.5 * (self.cross_entropy_loss(q1, scores2) + self.cross_entropy_loss(q2, scores1))

    @torch.no_grad()
    def sinkhorn(self, out, epsilon: float = 0.05, sinkhorn_iterations: int = 3):
        """Q is K-by-B for consistency with notations from the paper (out: B*K)"""
        Q = out  # - out.max() # remove comment to make it numerically more stable
        Q = torch.exp(Q / epsilon).t()
        Q /= torch.sum(Q)

        B = Q.shape[1]
        K = Q.shape[0]

        for _ in range(sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()

    def cross_entropy_loss(self, q, p):
        return torch.mean(torch.sum(q * F.log_softmax(p / self.temperature, dim=1), dim=1))