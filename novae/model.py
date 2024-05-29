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
from ._constants import EPS, INT_CONF, REPR, REPR_CORRECTED, SCORES, SWAV_CLASSES
from .data import NovaeDatamodule
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
        scgpt_model_dir: str | None = None,
        heads: int = 4,
        n_hops_local: int = 2,
        n_hops_ngh: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 10,
        out_channels: int = 64,
        batch_size: int = 512,
        lr: float = 1e-3,
        temperature: float = 0.1,
        num_prototypes: int = 1024,
        epoch_unfreeze_prototypes: int = 3,
        panel_dropout: float = 0.2,
        gene_expression_dropout: float = 0.1,
        background_noise_lambda: float = 5.0,
        sensitivity_noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.adatas, var_names = utils.prepare_adatas(adata, var_names=var_names)
        self.slide_key = slide_key

        self.save_hyperparameters(ignore=["adata", "slide_key"])

        ### Embeddings
        if scgpt_model_dir is None:
            self.genes_embedding = GenesEmbedding(var_names, embedding_size)
            self.genes_embedding.pca_init(self.adatas)
        else:
            self.genes_embedding = GenesEmbedding.from_scgpt_embedding(scgpt_model_dir)

        ### Modules
        self.backbone = GraphEncoder(
            self.genes_embedding.embedding_size, hidden_channels, num_layers, out_channels, heads
        )
        self.swav_head = SwavHead(out_channels, num_prototypes, temperature)
        self.augmentation = GraphAugmentation(
            panel_dropout, gene_expression_dropout, background_noise_lambda, sensitivity_noise_std
        )

        ### Losses
        self.bce_loss = nn.BCELoss()

        ### Datamodule
        if self.adatas is not None:
            self._datamodule = self.init_datamodule()

        ### Checkpoint
        self._checkpoint = None

    @property
    def datamodule(self) -> NovaeDatamodule:
        assert hasattr(self, "_datamodule"), "The datamodule was not initialized. Please provide an `adata` object."
        return self._datamodule

    def __repr__(self) -> str:
        return f"Novae model with {self.genes_embedding.voc_size} known genes\n   ├── [swav mode] {self.hparams.swav}\n   └── [checkpoint] {self._checkpoint}"

    def _embed_pyg_data(self, data: Data) -> Data:
        if self.training:
            data = self.augmentation(data)
        return self.genes_embedding(data)

    def _shuffle_pyg_data(self, data: Data) -> Data:
        shuffled_x = data.x[torch.randperm(data.x.size()[0])]
        return Data(
            x=shuffled_x, edge_index=data.edge_index, edge_attr=data.edge_attr, genes_indices=data.genes_indices
        )

    def training_step(self, batch: dict[str, Data], batch_idx: int):
        data_main = self._embed_pyg_data(batch["main"])

        if self.hparams.swav:
            data_ngh = self._embed_pyg_data(batch["neighbor"])

            x_main = self.backbone.node_x(data_main)
            x_ngh = self.backbone.node_x(data_ngh)

            loss = self.swav_head(x_main, x_ngh)
        else:
            data_shuffle = self._shuffle_pyg_data(data_main)

            x_main = self.backbone.edge_x(data_main)
            x_shuffle = self.backbone.edge_x(data_shuffle)

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

        return NovaeDatamodule(
            adatas,
            genes_embedding=self.genes_embedding,
            batch_size=self.hparams.batch_size,
            n_hops_local=self.hparams.n_hops_local,
            n_hops_ngh=self.hparams.n_hops_ngh,
        )

    def on_train_epoch_start(self):
        self.swav_head.prototypes.requires_grad = self.current_epoch >= self.hparams.epoch_unfreeze_prototypes

        self.datamodule.dataset.shuffle_obs_ilocs()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @torch.no_grad()
    def interaction_confidence(self, adata: AnnData | list[AnnData] | None = None) -> None:
        for adata in self.get_adatas(adata):
            datamodule = self.init_datamodule(adata)

            out = []
            for batch in utils.tqdm(datamodule.predict_dataloader()):
                batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
                data_main = self._embed_pyg_data(batch["main"])
                data_shuffle = self._shuffle_pyg_data(data_main)

                out.append(self.backbone.edge_x(data_main) - self.backbone.edge_x(data_shuffle))

            out = torch.concatenate(out)

            adata.obs[INT_CONF] = utils.fill_invalid_indices(out, adata, datamodule.dataset.valid_indices)

    @torch.no_grad()
    def swav_classes(self, adata: AnnData | list[AnnData] | None = None) -> None:
        for adata in self.get_adatas(adata):
            datamodule = self.init_datamodule(adata)

            out_rep = []
            out = []
            for batch in utils.tqdm(datamodule.predict_dataloader()):
                batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
                data_main = self._embed_pyg_data(batch["main"])
                x_main = self.backbone.node_x(data_main)
                out_ = F.normalize(x_main, dim=1, p=2)

                out_rep.append(out_)
                out.append(out_ @ self.swav_head.prototypes.T)

            out_rep = torch.cat(out_rep)
            out = torch.cat(out)
            out = self.swav_head.sinkhorn(out)

            adata.obsm[REPR] = utils.fill_invalid_indices(
                out_rep, adata, datamodule.dataset.valid_indices, dtype=np.float32
            )

            codes = utils.fill_invalid_indices(out, adata, datamodule.dataset.valid_indices, dtype=np.float32)
            adata.obs[SWAV_CLASSES] = np.where(np.isnan(codes).any(1), np.nan, np.argmax(codes, 1).astype(object))

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
            for batch in utils.tqdm(datamodule.predict_dataloader()):
                batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
                data_main = self._embed_pyg_data(batch["main"])
                edge_scores += self.backbone.edge_x(data_main, return_weights=True)

            adata.obsp[SCORES] = utils.fill_edge_scores(
                edge_scores,
                adata,
                datamodule.dataset.valid_indices,
                fill_value=np.nan,
                dtype=np.float32,
            )
            adata.write_h5ad("results/interactions/results_adata.h5ad")

    @torch.no_grad()
    def representation(self, adata: AnnData | list[AnnData] | None = None, return_res: bool = False) -> None:
        for adata in self.get_adatas(adata):
            datamodule = self.init_datamodule(adata)

            out = []
            for batch in utils.tqdm(datamodule.predict_dataloader()):
                batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
                data_main = self._embed_pyg_data(batch["main"])
                x_main = self.backbone.node_x(data_main)
                out_ = F.normalize(x_main, dim=1, p=2)
                out.append(out_)

            out = torch.concat(out, dim=0)

            if return_res:
                return out

            adata.obsm[REPR] = utils.fill_invalid_indices(out, adata, datamodule.dataset.valid_indices)

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
        model = cls.load_from_checkpoint(artifact_dir / "model.ckpt", strict=False, **kwargs)
        model._checkpoint = f"wandb: {name}"
        return model

    def _get_centroids(self, adata: AnnData, domains: list[str], obs_key: str) -> tuple[np.ndarray, np.ndarray]:
        centroids, is_valid = [], []

        for d in domains:
            where = adata.obs[obs_key] == d
            if where.any():
                centroids.append(np.mean(adata.obsm[REPR][where], axis=0))
                is_valid.append(True)
            else:
                centroids.append(np.ones(adata.obsm[REPR].shape[1]))
                is_valid.append(False)

        centroids = np.stack(centroids)
        is_valid = np.array(is_valid)
        centroids /= np.linalg.norm(centroids, ord=2, axis=-1, keepdims=True) + EPS

        return centroids, is_valid

    def batch_effect_correction(
        self, adata: AnnData | list[AnnData] | None, obs_key: str, index_reference: int | None = None
    ):
        adatas = self.get_adatas(adata)

        if index_reference is None:
            index_reference = max(range(len(adatas)), key=lambda i: adatas[i].n_obs)

        adata_ref = adatas[index_reference]
        adata_ref.obsm[REPR_CORRECTED] = adata_ref.obsm[REPR]

        ref_domains = adata_ref.obs[obs_key]
        domains = list(np.unique(ref_domains))
        centroids_reference, is_valid_ref = self._get_centroids(adata_ref, domains, obs_key)

        if not is_valid_ref.all():
            log.warn("Not all domains found in the reference, which may lead to a bad batch effect correction.")

        for i, adata in enumerate(adatas):
            if i == index_reference:
                continue

            centroids, is_valid = self._get_centroids(adata, domains, obs_key)
            rotations = self.swav_head.rotations_geodesic(centroids, centroids_reference)

            adata.obsm[REPR_CORRECTED] = np.zeros_like(adata.obsm[REPR])

            for j, d in enumerate(domains):
                if not (is_valid[j] and is_valid_ref[j]):
                    continue

                where = adata.obs[obs_key] == d

                coords = adata.obsm[REPR][where]
                coords = (rotations[j] @ coords.T).T
                coords /= np.linalg.norm(coords, ord=2, axis=-1, keepdims=True) + EPS

                adata.obsm[REPR_CORRECTED][where] = coords

    def fit(self, adata: AnnData | list[AnnData], **kwargs):
        self.adatas, _ = utils.prepare_adatas(adata, var_names=self.genes_embedding.vocabulary)
        self._datamodule = self.init_datamodule()

        trainer = L.Trainer(**kwargs)
        trainer.fit(self, datamodule=self.datamodule)
