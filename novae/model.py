from __future__ import annotations

import logging

import lightning as L
import numpy as np
import torch
from anndata import AnnData
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.data import Data

from . import __version__, utils
from ._constants import Keys, Nums
from .data import NovaeDatamodule
from .module import CellEmbedder, GraphAugmentation, GraphEncoder, SwavHead

log = logging.getLogger(__name__)


class Novae(L.LightningModule):
    def __init__(
        self,
        adata: AnnData | list[AnnData] | None = None,
        slide_key: str = None,
        var_names: list[str] = None,
        scgpt_model_dir: str | None = None,
        embedding_size: int = 100,
        n_hops_local: int = 2,
        n_hops_ngh: int = 2,
        heads: int = 4,
        hidden_size: int = 64,
        num_layers: int = 10,
        output_size: int = 64,
        batch_size: int = 512,
        lr: float = 1e-3,
        temperature: float = 0.5,
        num_prototypes: int = 256,
        panel_subset_size: float = 0.6,
        background_noise_lambda: float = 8.0,
        sensitivity_noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.slide_key = slide_key

        if scgpt_model_dir is None:
            self.adatas, var_names = utils.prepare_adatas(adata, var_names=var_names)
            self.cell_embedder = CellEmbedder(var_names, embedding_size)
            self.cell_embedder.pca_init(self.adatas)
        else:
            self.cell_embedder = CellEmbedder.from_scgpt_embedding(scgpt_model_dir)
            self.hparams["embedding_size"] = self.cell_embedder.embedding_size
            self.adatas, var_names = utils.prepare_adatas(adata, var_names=self.cell_embedder.gene_names)

        self.save_hyperparameters(ignore=["adata", "slide_key", "scgpt_model_dir"])

        ### Modules
        self.backbone = GraphEncoder(self.cell_embedder.embedding_size, hidden_size, num_layers, output_size, heads)
        self.swav_head = SwavHead(output_size, num_prototypes, temperature)
        self.augmentation = GraphAugmentation(panel_subset_size, background_noise_lambda, sensitivity_noise_std)

        ### Losses
        self.bce_loss = nn.BCELoss()

        ### Datamodule
        if self.adatas is not None:
            self._datamodule = self._init_datamodule()

        ### Checkpoint
        self._checkpoint = None

    @property
    def datamodule(self) -> NovaeDatamodule:
        assert hasattr(self, "_datamodule"), "The datamodule was not initialized. Please provide an `adata` object."
        return self._datamodule

    def __repr__(self) -> str:
        return f"Novae model with {self.cell_embedder.voc_size} known genes\n   └── [checkpoint] {self._checkpoint}"

    def _embed_pyg_data(self, data: Data) -> Data:
        if self.training:
            data = self.augmentation(data)
        return self.cell_embedder(data)

    def forward(self, batch: dict[str, Data]) -> torch.Tensor:
        return {key: self.backbone(self._embed_pyg_data(data)) for key, data in batch.items()}

    def training_step(self, batch: dict[str, Data], batch_idx: int):
        out: dict[str, Data] = self(batch)

        loss = self.swav_head(out["main"], out["neighbor"])

        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=True,
            batch_size=self.hparams.batch_size,
            prog_bar=True,
        )

        return loss

    def _init_datamodule(self, adata: AnnData | list[AnnData] | None = None):
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
            cell_embedder=self.cell_embedder,
            batch_size=self.hparams.batch_size,
            n_hops_local=self.hparams.n_hops_local,
            n_hops_ngh=self.hparams.n_hops_ngh,
        )

    def on_train_epoch_start(self):
        self.datamodule.dataset.shuffle_obs_ilocs()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @torch.no_grad()
    def latent_representation(self, adata: AnnData | list[AnnData] | None = None) -> None:
        for adata in self._get_adatas(adata):
            datamodule = self._init_datamodule(adata)

            out_rep = []
            out = []
            for batch in utils.tqdm(datamodule.predict_dataloader()):
                batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
                data_main = self._embed_pyg_data(batch["main"])
                x_main = self.backbone(data_main)
                out_rep.append(x_main)
                out_ = F.normalize(x_main, dim=1, p=2)

                out.append(out_ @ self.swav_head.prototypes.T)

            out_rep = torch.cat(out_rep)
            out = torch.cat(out)
            out = self.swav_head.sinkhorn(out)

            adata.obsm[Keys.REPR] = utils.fill_invalid_indices(
                out_rep, adata.n_obs, datamodule.dataset.valid_indices[0], fill_value=0
            )

            codes = utils.fill_invalid_indices(out, adata.n_obs, datamodule.dataset.valid_indices[0])
            adata.obs[Keys.SWAV_CLASSES] = np.where(np.isnan(codes).any(1), np.nan, np.argmax(codes, 1).astype(object))

    def _get_adatas(self, adata: AnnData | list[AnnData] | None):
        if adata is None:
            return self.adatas
        return utils.prepare_adatas(adata, var_names=self.cell_embedder.gene_names)[0]

    def assign_domains(self, adata: AnnData, k: int, key_added: str | None = None) -> str:
        if key_added is None:
            key_added = f"{Keys.NICHE_PREFIX}{k}"
        adata.obs[key_added] = self.swav_head.map_leaves_domains(adata.obs[Keys.SWAV_CLASSES], k)
        log.info(f"Spatial domains saved in `adata.obs['{key_added}']`")
        return key_added

    @classmethod
    def load_from_wandb_artifact(cls, name: str, **kwargs) -> "Novae":
        artifact_dir = utils._load_wandb_artifact(name)

        try:
            model = cls.load_from_checkpoint(artifact_dir / "model.ckpt", strict=False, **kwargs)
        except:
            ckpt_version = torch.load(artifact_dir / "model.ckpt").get(Keys.NOVAE_VERSION, "unknown")
            raise ValueError(f"The model was trained with `novae=={ckpt_version}`, but your version is {__version__}")

        model._checkpoint = f"wandb: {name}"
        return model

    def _get_centroids(self, adata: AnnData, domains: list[str], obs_key: str) -> tuple[np.ndarray, np.ndarray]:
        centroids, is_valid = [], []

        for d in domains:
            where = adata.obs[obs_key] == d
            if where.any():
                centroids.append(np.mean(adata.obsm[Keys.REPR][where], axis=0))
                is_valid.append(True)
            else:
                centroids.append(np.ones(adata.obsm[Keys.REPR].shape[1]))
                is_valid.append(False)

        centroids = np.stack(centroids)
        is_valid = np.array(is_valid)
        centroids /= np.linalg.norm(centroids, ord=2, axis=-1, keepdims=True) + Nums.EPS

        return centroids, is_valid

    def batch_effect_correction(
        self, adata: AnnData | list[AnnData] | None, obs_key: str, index_reference: int | None = None
    ):
        adatas = self._get_adatas(adata)

        if index_reference is None:
            index_reference = max(range(len(adatas)), key=lambda i: adatas[i].n_obs)

        adata_ref = adatas[index_reference]
        adata_ref.obsm[Keys.REPR_CORRECTED] = adata_ref.obsm[Keys.REPR]

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

            adata.obsm[Keys.REPR_CORRECTED] = np.zeros_like(adata.obsm[Keys.REPR])

            for j, d in enumerate(domains):
                if not (is_valid[j] and is_valid_ref[j]):
                    continue

                where = adata.obs[obs_key] == d

                coords = adata.obsm[Keys.REPR][where]
                coords = (rotations[j] @ coords.T).T
                coords /= np.linalg.norm(coords, ord=2, axis=-1, keepdims=True) + Nums.EPS

                adata.obsm[Keys.REPR_CORRECTED][where] = coords

    def on_save_checkpoint(self, checkpoint):
        checkpoint[Keys.NOVAE_VERSION] = __version__

    def train(self, adata: AnnData | list[AnnData] | None, **kwargs):
        if adata is not None:
            self.adatas, _ = utils.prepare_adatas(adata, var_names=self.cell_embedder.gene_names)
            self._datamodule = self._init_datamodule()

        trainer = L.Trainer(**kwargs)
        trainer.fit(self, datamodule=self.datamodule)
