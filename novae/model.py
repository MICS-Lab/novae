from __future__ import annotations

import logging

import lightning as L
import numpy as np
import torch
from anndata import AnnData
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch_geometric.data import Data

from . import __version__, utils
from ._constants import Keys, Nums
from .data import NovaeDatamodule, NovaeDataset
from .module import CellEmbedder, GraphAugmentation, GraphEncoder, SwavHead

log = logging.getLogger(__name__)


class Novae(L.LightningModule):
    """Novae model class. It can be used to load a pretrained model or train a new one.

    !!! note "Example usage"
        ```python
        import novae

        model = novae.Novae(adata)

        model.fit()
        model.compute_representation()
        model.assign_domains()
        ```
    """

    @utils.format_docs
    def __init__(
        self,
        adata: AnnData | list[AnnData] | None = None,
        slide_key: str = None,
        scgpt_model_dir: str | None = None,
        var_names: list[str] = None,
        embedding_size: int = 100,
        output_size: int = 64,
        n_hops_local: int = 2,
        n_hops_view: int = 2,
        heads: int = 4,
        hidden_size: int = 64,
        num_layers: int = 10,
        batch_size: int = 512,
        lr: float = 1e-3,
        temperature: float = 0.1,
        num_prototypes: int = 256,
        panel_subset_size: float = 0.6,
        background_noise_lambda: float = 8.0,
        sensitivity_noise_std: float = 0.05,
        epoch_unfreeze_prototypes: int = 2,
    ) -> None:
        """

        Args:
            {adata}
            {slide_key}
            {scgpt_model_dir}
            {var_names}
            embedding_size: Size of the gene embedding. Do not use it when loading embeddings from scGPT.
            output_size: Size of the latent space.
            {n_hops_local}
            {n_hops_view}
            heads: Number of heads for the graph encoder.
            hidden_size: Hidden size for the graph encoder.
            num_layers: Number of layers for the graph encoder.
            batch_size: Mini-batch size.
            lr: Model learning rate.
            temperature: Swav temperature.
            num_prototypes: Number of prototypes, or "leaves" niches.
            {panel_subset_size}
            {background_noise_lambda}
            {sensitivity_noise_std}
        """
        super().__init__()
        self.slide_key = slide_key

        if scgpt_model_dir is None:
            self.adatas, var_names = utils.prepare_adatas(adata, slide_key=slide_key, var_names=var_names)
            self.cell_embedder = CellEmbedder(var_names, embedding_size)
            self.cell_embedder.pca_init(self.adatas)
        else:
            self.cell_embedder = CellEmbedder.from_scgpt_embedding(scgpt_model_dir)
            embedding_size = self.cell_embedder.embedding_size
            _scgpt_var_names = self.cell_embedder.gene_names
            self.adatas, var_names = utils.prepare_adatas(adata, slide_key=slide_key, var_names=_scgpt_var_names)

        self.save_hyperparameters(ignore=["adata", "slide_key", "scgpt_model_dir"])

        ### Modules
        self.encoder = GraphEncoder(self.cell_embedder.embedding_size, hidden_size, num_layers, output_size, heads)
        self.swav_head = SwavHead(output_size, num_prototypes, temperature)
        self.augmentation = GraphAugmentation(panel_subset_size, background_noise_lambda, sensitivity_noise_std)

        ### Losses
        self.bce_loss = nn.BCELoss()

        ### Misc
        self._num_workers = 0
        self._checkpoint = None
        self._datamodule = None

    @property
    def datamodule(self) -> NovaeDatamodule:
        assert (
            self._datamodule is not None
        ), "The datamodule was not initialized. You first need to fit the model, i.e. `model.fit(...)`"
        return self._datamodule

    @property
    def dataset(self) -> NovaeDataset:
        return self.datamodule.dataset

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value: int) -> None:
        self._num_workers = value
        if self._datamodule is not None:
            self._datamodule.num_workers = value

    def __repr__(self) -> str:
        info_dict = {
            "known genes": self.cell_embedder.voc_size,
            "parameters": utils.pretty_num_parameters(self),
            "checkpoint": self._checkpoint,
        }
        rows = ["Novae model"] + [f"[{k}]: {v}" for k, v in info_dict.items()]
        return "\n   ├── ".join(rows[:-1]) + "\n   └── " + rows[-1]

    def _embed_pyg_data(self, data: Data) -> Data:
        if self.training:
            data = self.augmentation(data)
        return self.cell_embedder(data)

    def forward(self, batch: dict[str, Data]) -> dict[str, Tensor]:
        return {key: self.encoder(self._embed_pyg_data(data)) for key, data in batch.items()}

    def training_step(self, batch: dict[str, Data], batch_idx: int):
        out: dict[str, Data] = self(batch)

        loss, mean_entropy_normalized = self.swav_head(out["main"], out["view"])

        self._log_all({"loss": loss, "entropy": mean_entropy_normalized})

        return loss

    def _log_all(self, log_dict: dict[str, float], **kwargs):
        for name, value in log_dict.items():
            self.log(
                f"train/{name}",
                value,
                on_epoch=True,
                on_step=True,
                batch_size=self.hparams.batch_size,
                prog_bar=True,
                **kwargs,
            )

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
            n_hops_view=self.hparams.n_hops_view,
            num_workers=self._num_workers,
        )

    def on_train_epoch_start(self):
        self.training = True
        self.swav_head.prototypes.requires_grad_(self.current_epoch >= self.hparams.epoch_unfreeze_prototypes)
        self.datamodule.dataset.shuffle_obs_ilocs()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    @utils.format_docs
    @torch.no_grad()
    def compute_representation(
        self,
        adata: AnnData | list[AnnData] | None = None,
        slide_key: str | None = None,
        accelerator: str = "cpu",
        num_workers: int | None = None,
    ) -> None:
        """Compute the latent representation of Novae for all cells neighborhoods.

        Note:
            Representations are saved in `adata.obsm["novae_latent"]`

        Args:
            {adata}
            {slide_key}
            {accelerator}
            num_workers: Number of workers for the dataloader.
        """
        self.training = False
        device = self._parse_hardware_args(accelerator, num_workers, return_device=True)
        self.to(device)

        if adata is None and len(self.adatas) == 1:  # using existing datamodule
            self._compute_representation_datamodule(self.adatas[0], self.datamodule)
            return

        for adata in self._get_adatas(adata, slide_key=slide_key):
            datamodule = self._init_datamodule(adata)
            self._compute_representation_datamodule(adata, datamodule)

    def _compute_representation_datamodule(self, adata: AnnData, datamodule: NovaeDatamodule):
        valid_indices = datamodule.dataset.valid_indices[0]

        representations = []
        codes = []
        for batch in utils.tqdm(datamodule.predict_dataloader()):
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            representation = self.encoder(self._embed_pyg_data(batch["main"]))

            representations.append(representation)
            repr_normalized = F.normalize(representation, dim=1, p=2)
            codes.append(repr_normalized @ self.swav_head.prototypes.T)

        representations = torch.cat(representations)
        adata.obsm[Keys.REPR] = utils.fill_invalid_indices(representations, adata.n_obs, valid_indices, fill_value=0)

        codes = self._apply_sinkhorn_per_slide(torch.cat(codes), adata, valid_indices)
        codes = utils.fill_invalid_indices(codes, adata.n_obs, valid_indices)
        adata.obs[Keys.SWAV_CLASSES] = np.where(np.isnan(codes).any(1), np.nan, np.argmax(codes, 1).astype(object))

        return representations, codes, valid_indices

    def _apply_sinkhorn_per_slide(self, scores: Tensor, adata: AnnData, valid_indices: np.ndarray) -> Tensor:
        slide_ids = adata.obs[Keys.SLIDE_ID].values[valid_indices]

        unique_slide_ids = np.unique(slide_ids)

        if len(unique_slide_ids) == 1:
            return self.swav_head.sinkhorn(scores)

        for slide_id in unique_slide_ids:
            indices = np.where(slide_ids == slide_id)[0]
            scores[indices] = self.swav_head.sinkhorn(scores[indices])

        return scores

    def _get_adatas(self, adata: AnnData | list[AnnData] | None, slide_key: str | None = None):
        if adata is None:
            return self.adatas
        return utils.prepare_adatas(adata, slide_key=slide_key, var_names=self.cell_embedder.gene_names)[0]

    @utils.format_docs
    def assign_domains(
        self, adata: AnnData | list[AnnData] | None = None, k: int = 10, key_added: str | None = None
    ) -> str:
        """Assign a domain (or niche) to each cell based on the "leaves" classes.

        Note:
            You'll need to run `model.compute_representation(...)` first.

            The domains are saved in `adata.obs["novae_niche_k"]` by default, where `k=10`.

        Args:
            {adata}
            k: Number of domains (or niches) to assign.
            key_added: The spatial domains will be saved in `adata.obs[key_added]`. By default, it is `"novae_niche_k"`.

        Returns:
            The name of the key added to `adata.obs`.
        """
        if key_added is None:
            key_added = f"{Keys.NICHE_PREFIX}{k}"

        adatas = [adata] if isinstance(adata, AnnData) else (adata if isinstance(adata, list) else self.adatas)

        for adata in adatas:
            assert (
                Keys.SWAV_CLASSES in adata.obs
            ), f"Did not found `adata.obs['{Keys.SWAV_CLASSES}']`. Please run `model.compute_representation(...)` first"
            adata.obs[key_added] = self.swav_head.map_leaves_domains(adata.obs[Keys.SWAV_CLASSES], k)

        log.info(f"Spatial domains saved in `adata.obs['{key_added}']`")
        return key_added

    @classmethod
    def load_from_wandb_artifact(cls, name: str, map_location: str = "cpu", **kwargs) -> "Novae":
        """Initialize a model from a Weights & Biases artifact.

        Args:
            name: Name of the artifact.
            map_location: If your checkpoint saved a GPU model and you now load on CPUs or a different number of GPUs, use this to map to the new setup. The behaviour is the same as in `torch.load()`.
            kwargs: Optional kwargs for the Pytorch Lightning `load_from_checkpoint` method.

        Returns:
            A Novae model.
        """
        artifact_path = utils._load_wandb_artifact(name) / "model.ckpt"

        try:
            model = cls.load_from_checkpoint(artifact_path, map_location=map_location, strict=False, **kwargs)
        except:
            ckpt_version = torch.load(artifact_path, map_location=map_location).get(Keys.NOVAE_VERSION, "unknown")
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
        self,
        adata: AnnData | list[AnnData] | None,
        obs_key: str,
        slide_key: str | None = None,
        index_reference: int | None = None,
    ):
        adatas = self._get_adatas(adata, slide_key=slide_key)

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

    def _parse_hardware_args(self, accelerator: str, num_workers: int | None, return_device: bool = False):
        if accelerator == "cpu" and num_workers:
            log.warn(
                f"Setting `num_workers != 0` with {accelerator=} can be very slow. Consider using a GPU, or setting `num_workers=0`."
            )

        if num_workers is not None:
            self.num_workers = num_workers

        if return_device:
            return utils.parse_device_args(accelerator)

    @utils.format_docs
    def fit(
        self,
        adata: AnnData | list[AnnData] | None = None,
        slide_key: str | None = None,
        max_epochs: int = 20,
        accelerator: str = "cpu",
        num_workers: int | None = None,
        min_delta: float = 0.1,
        patience: int = 3,
        callbacks: list[Callback] | None = None,
        enable_checkpointing: bool = False,
        logger: Logger | list[Logger] | bool = False,
        **kwargs,
    ):
        """Train a Novae model. The training will be stopped by early stopping.

        Args:
            {adata}
            {slide_key}
            max_epochs: Maximum number of training epochs.
            {accelerator}
            num_workers: Number of workers for the dataloader.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement (early stopping).
            patience: Number of epochs with no improvement after which training will be stopped (early stopping).
            callbacks: Optional list of Pytorch lightning callbacks.
            enable_checkpointing: Whether to enable model checkpointing.
            logger: The pytorch lightning logger.
            kwargs: Optional kwargs for the Pytorch Lightning `Trainer` class.
        """
        self._parse_hardware_args(accelerator, num_workers)

        if adata is not None:
            self.adatas, _ = utils.prepare_adatas(adata, slide_key=slide_key, var_names=self.cell_embedder.gene_names)

        self._datamodule = self._init_datamodule()

        early_stopping = EarlyStopping(
            monitor="train/loss_epoch",
            min_delta=min_delta,
            patience=patience,
            check_on_train_epoch_end=True,
        )
        callbacks = [early_stopping] + (callbacks or [])
        enable_checkpointing = enable_checkpointing or any(isinstance(c, ModelCheckpoint) for c in callbacks)

        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            callbacks=callbacks,
            logger=logger,
            enable_checkpointing=enable_checkpointing,
            **kwargs,
        )
        trainer.fit(self, datamodule=self.datamodule)
