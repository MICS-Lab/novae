from __future__ import annotations

import logging

import lightning as L
import numpy as np
import torch
from anndata import AnnData
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.data import Data

from . import utils
from ._constants import CODES, INT_CONF, QC, REPR, SCORES, SWAV_CLASSES
from .data import LocalAugmentationDatamodule
from .module import GenesEmbedding, GraphAugmentation, GraphEncoder, SwavHead

log = logging.getLogger(__name__)


class Novae(L.LightningModule):
    def __init__(
        self,
        adata: AnnData | list[AnnData] | None = None,
        swav: bool = True,
        slide_key: str = None,
        var_names: list[str] = None,
        embedding_size: int = 100,
        heads: int = 4,
        n_hops: int = 2,
        n_intermediate: int = 4,
        hidden_channels: int = 64,
        num_layers: int = 10,
        out_channels: int = 64,
        batch_size: int = 512,
        lr: float = 1e-3,
        temperature: float = 0.1,
        num_prototypes: int = 1024,
        epoch_unfreeze_prototypes: int = 10,
    ) -> None:
        super().__init__()
        self.adatas, var_names = utils.prepare_adatas(adata, var_names=var_names)
        self.slide_key = slide_key

        self.save_hyperparameters(ignore=["adata", "slide_key"])

        ### Embeddings
        self.genes_embedding = GenesEmbedding(var_names, embedding_size)
        self.genes_embedding.pca_init(self.adatas)

        ### Modules
        self.backbone = GraphEncoder(embedding_size, hidden_channels, num_layers, out_channels, heads)
        self.swav_head = SwavHead(out_channels, num_prototypes, temperature)
        self.augmentation = GraphAugmentation()

        ### Losses
        self.bce_loss = nn.BCELoss()

        ### Datamodule
        if self.adatas is not None:
            self._datamodule = self.init_datamodule()

        ### Checkpoint
        self._checkpoint = None

    @property
    def datamodule(self) -> LocalAugmentationDatamodule:
        assert hasattr(self, "_datamodule"), "The datamodule was not initialized. Please provide an `adata` object."
        return self._datamodule

    def __repr__(self) -> str:
        return f"Novae model with {self.genes_embedding.voc_size} known genes\n   ├── [swav mode] {self.hparams.swav}\n   └── [checkpoint] {self._checkpoint}"

    def on_train_start(self):
        # self.init_prototypes()
        ...  # TODO: remove?

    def init_prototypes(self, method="sample"):
        # TODO: do we keep it?
        X = self.representation(self.adatas[0], return_res=True)

        if method == "kmeans":
            self.swav_head.init_prototypes_kmeans(X)
        if method == "sample":
            self.swav_head.init_prototypes_sample(X)

    def training_step(self, batch: tuple[Data, Data, Data], batch_idx: int):
        if self.hparams.swav:
            x_main = self.backbone.node_x(batch[0])
            x_ngh = self.backbone.node_x(batch[2])

            loss = self.swav_head(x_main, x_ngh)
        else:
            x_main = self.backbone.edge_x(batch[0])
            x_shuffle = self.backbone.edge_x(batch[1])

            loss = self.bce_loss(x_main, torch.ones_like(x_main, device=x_main.device))
            loss += self.bce_loss(x_shuffle, torch.zeros_like(x_shuffle, device=x_shuffle.device))

        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=True,
            batch_size=self.hparams.batch_size,
            prog_bar=True,
        )

        return loss

    def init_datamodule(self, adata: AnnData | list[AnnData] | None = None):
        if adata is None:
            adatas = self.adatas
        elif isinstance(adata, AnnData):
            adatas = [adata]
        elif isinstance(adata, list):
            adatas = adata
        else:
            raise ValueError(f"Invalid type {type(adata)} for argument adata")

        return LocalAugmentationDatamodule(
            adatas,
            self.genes_embedding,
            augmentation=self.augmentation,
            batch_size=self.hparams.batch_size,
            n_hops=self.hparams.n_hops,
            n_intermediate=self.hparams.n_intermediate,
        )

    def on_train_epoch_start(self):
        self.swav_head.prototypes.requires_grad = self.current_epoch >= self.hparams.epoch_unfreeze_prototypes

        self.datamodule.shuffle_obs_ilocs()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @torch.no_grad()
    def interaction_confidence(self, adata: AnnData | list[AnnData] | None = None) -> None:
        for adata in self.get_adatas(adata):
            datamodule = self.init_datamodule(adata)

            out = torch.concatenate(
                [
                    self.backbone.edge_x(batch[0]) - self.backbone.edge_x(batch[1])
                    for batch in utils.tqdm(datamodule.predict_dataloader())
                ]
            )

            adata.obs[INT_CONF] = utils.fill_invalid_indices(out, adata, datamodule.valid_indices)

    @torch.no_grad()
    def swav_classes(self, adata: AnnData | list[AnnData] | None = None) -> None:
        for adata in self.get_adatas(adata):
            assert CODES in adata.obsm, "Codes are not computed. Run model.codes() first."

            codes = adata.obsm[CODES]
            adata.obs[SWAV_CLASSES] = np.where(np.isnan(codes).any(1), np.nan, np.argmax(codes, 1).astype(object))

    def predict_step(self, batch):
        data_main, *_ = batch
        x_main = self.backbone.node_x(data_main)
        out = F.normalize(x_main, dim=1, p=2)
        return out @ self.swav_head.prototypes

    @torch.no_grad()
    def codes(
        self,
        adata: AnnData | list[AnnData] | None = None,
        sinkhorn: bool = True,
    ) -> None:
        for adata in self.get_adatas(adata):
            datamodule = self.init_datamodule(adata)

            out_rep = []
            out = []
            for data_main, *_ in utils.tqdm(datamodule.predict_dataloader()):
                x_main = self.backbone.node_x(data_main)
                out_ = F.normalize(x_main, dim=1, p=2)

                out_rep.append(out_)
                out.append(out_ @ self.swav_head.prototypes)

            out_rep = torch.cat(out_rep)
            out = torch.cat(out)

            if sinkhorn:
                out = self.swav_head.sinkhorn(out)

            adata.obsm[REPR] = utils.fill_invalid_indices(out_rep, adata, datamodule.valid_indices, dtype=np.float32)
            adata.obsm[CODES] = utils.fill_invalid_indices(out, adata, datamodule.valid_indices, dtype=np.float32)

    @torch.no_grad()
    def edge_scores(self, adata: AnnData | list[AnnData] | None = None) -> None:
        """
        Computes and assigns edge scores to the given AnnData object(s).

        This method processes either a single AnnData object, a list of AnnData objects, or all available
        AnnData objects, to compute edge scores for each based on the model's backbone predictions.
        The computed edge scores are then stored within the `obsp` attribute of each respective AnnData
        object under a predefined key. After computation, the updated AnnData object(s) are saved to disk.

        Parameters:
        - adata (Union[AnnData, List[AnnData], None], optional): The AnnData object(s) to process. If None,
          the method retrieves available AnnData objects using the `get_adatas` method. Default is None.

        Returns:
        - None: The AnnData object(s) are updated in-place and saved to disk. The computed edge scores
          are stored within the `obsp` attribute of each AnnData object.
        """
        for adata in self.get_adatas(adata):
            datamodule = self.init_datamodule(adata)

            edge_scores = []
            for data_main, *_ in utils.tqdm(datamodule.predict_dataloader()):
                edge_scores += self.backbone.edge_x(data_main, return_weights=True)

            adata.obsp[SCORES] = utils.fill_edge_scores(
                edge_scores,
                adata,
                datamodule.valid_indices,
                fill_value=np.nan,
                dtype=np.float32,
            )
            adata.write_h5ad("results/interactions/results_adata.h5ad")

    @torch.no_grad()
    def representation(self, adata: AnnData | list[AnnData] | None = None, return_res: bool = False) -> None:
        for adata in self.get_adatas(adata):
            datamodule = self.init_datamodule(adata)

            out = []
            for data_main, *_ in utils.tqdm(datamodule.predict_dataloader()):
                x_main = self.backbone.node_x(data_main)
                out_ = F.normalize(x_main, dim=1, p=2)
                out.append(out_)

            out = torch.concat(out, dim=0)

            if return_res:
                return out

            adata.obsm[REPR] = utils.fill_invalid_indices(out, adata, datamodule.valid_indices)

    def get_adatas(self, adata: AnnData | list[AnnData] | None):
        if adata is None:
            return self.adatas
        return utils.prepare_adatas(adata, var_names=self.genes_embedding.vocabulary)[0]

    def assign_domains(self, adata: AnnData, k: int, key_added: str | None = None) -> str:
        if key_added is None:
            key_added = f"{SWAV_CLASSES}_{k}"
        adata.obs[key_added] = self.swav_head.assign_classes_level(adata.obs[SWAV_CLASSES], k)
        log.info(f"Spatial domains saved in `adata.obs['{key_added}']`")
        return key_added

    @classmethod
    def load_from_wandb_artifact(cls, name: str, **kwargs) -> "Novae":
        artifact_dir = utils._load_wandb_artifact(name)
        model = cls.load_from_checkpoint(artifact_dir / "model.ckpt", **kwargs)
        model._checkpoint = f"wandb: {name}"
        return model
