from __future__ import annotations

import logging

import lightning as L
import numpy as np
import torch
from anndata import AnnData
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
from torch import Tensor, optim
from torch_geometric.data import Data

from . import __version__, plot, utils
from ._constants import Keys, Nums
from .data import NovaeDatamodule, NovaeDataset
from .module import CellEmbedder, GraphAugmentation, GraphEncoder, SwavHead

log = logging.getLogger(__name__)


class Novae(L.LightningModule, PyTorchModelHubMixin):
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
        var_names: list[str] | None = None,
        tissue_names: list[str] | None = None,
        embedding_size: int = 100,
        output_size: int = 64,
        n_hops_local: int = 2,
        n_hops_view: int = 2,
        heads: int = 4,
        hidden_size: int = 64,
        num_layers: int = 10,
        batch_size: int = 512,
        temperature: float = 0.1,
        temperature_weight_proto: float = 0.05,
        num_prototypes: int = 256,
        panel_subset_size: float = 0.6,
        background_noise_lambda: float = 8.0,
        sensitivity_noise_std: float = 0.05,
        lambda_regularization: float = 0.0,
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
            temperature: Swav temperature.
            num_prototypes: Number of prototypes, or "leaves" niches.
            {panel_subset_size}
            {background_noise_lambda}
            {sensitivity_noise_std}
        """
        super().__init__()
        self.slide_key = slide_key

        ### Initialize cell embedder and prepare adata(s) object(s)
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
        self.mode = utils.Mode()

        ### Initialize modules
        self.encoder = GraphEncoder(embedding_size, hidden_size, num_layers, output_size, heads)
        self.augmentation = GraphAugmentation(panel_subset_size, background_noise_lambda, sensitivity_noise_std)
        self.swav_head = SwavHead(
            self.mode, output_size, num_prototypes, temperature, temperature_weight_proto, lambda_regularization
        )
        self._init_tissue_queue(tissue_names)

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
        return utils.pretty_model_repr(info_dict)

    def _embed_pyg_data(self, data: Data) -> Data:
        if self.training:
            data = self.augmentation(data)
        return self.cell_embedder(data)

    def forward(self, batch: dict[str, Data]) -> dict[str, Tensor]:
        return {key: self.encoder(self._embed_pyg_data(data)) for key, data in batch.items()}

    def training_step(self, batch: dict[str, Data], batch_idx: int):
        out: dict[str, Data] = self(batch)
        tissue = batch["main"].get(Keys.UNS_TISSUE, [None])[0]

        loss, mean_entropy_normalized = self.swav_head(out["main"], out["view"], tissue)

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

    def _init_tissue_queue(self, tissue_names: list[str] | None):
        if tissue_names is not None:
            self.swav_head.init_queue(tissue_names)
            return

        if self.adatas is None or Keys.UNS_TISSUE not in self.adatas[0].uns:
            return

        tissue_names = list({adata.uns[Keys.UNS_TISSUE] for adata in self.adatas})
        if len(tissue_names) > 1:
            self.mode.queue_mode = True
            self.hparams["tissue_names"] = tissue_names
            self.swav_head.init_queue(tissue_names)

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

        self.datamodule.dataset.shuffle_obs_ilocs()

        after_warm_up = self.current_epoch >= Nums.WARMUP_EPOCHS

        self.swav_head.prototypes.requires_grad_(after_warm_up or not self.mode.freeze_mode)
        self.mode.use_queue = after_warm_up and self.mode.queue_mode

    def configure_optimizers(self):
        lr = self._lr if hasattr(self, "_lr") else 1e-3
        return optim.Adam(self.parameters(), lr=lr)

    @utils.format_docs
    @utils.requires_fit
    @torch.no_grad()
    def compute_representation(
        self,
        adata: AnnData | list[AnnData] | None = None,
        slide_key: str | None = None,
        zero_shot: bool = False,
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
        self.mode.zero_shot = zero_shot
        self.training = False
        device = self._parse_hardware_args(accelerator, num_workers, return_device=True)
        self.to(device)

        if adata is None and len(self.adatas) == 1:  # using existing datamodule
            adatas = self.adatas
            self._compute_representation_datamodule(self.adatas[0], self.datamodule)
        else:
            adatas = self._get_adatas(adata, slide_key=slide_key)
            for adata in adatas:
                datamodule = self._init_datamodule(adata)
                self._compute_representation_datamodule(adata, datamodule)

        if self.mode.zero_shot:
            latent = np.concatenate([adata.obsm[Keys.REPR][utils.get_valid_indices(adata)] for adata in adatas])
            self.swav_head.set_kmeans_prototypes(latent)

            for adata in adatas:
                self._compute_leaves(adata, None, None)

    def _compute_representation_datamodule(self, adata: AnnData, datamodule: NovaeDatamodule):
        valid_indices = datamodule.dataset.valid_indices[0]
        representations, scores = [], []

        for batch in utils.tqdm(datamodule.predict_dataloader()):
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            batch_repr = self.encoder(self._embed_pyg_data(batch["main"]))

            representations.append(batch_repr)

            if not self.mode.zero_shot:
                batch_scores = self.swav_head.compute_scores(batch_repr)
                scores.append(batch_scores)

        representations = torch.cat(representations)
        adata.obsm[Keys.REPR] = utils.fill_invalid_indices(representations, adata.n_obs, valid_indices, fill_value=0)

        if not self.mode.zero_shot:
            scores = torch.cat(scores)
            self._compute_leaves(adata, scores, valid_indices)

    def _compute_leaves(self, adata: AnnData, scores: Tensor | None, valid_indices: np.ndarray | None):
        assert (scores is None) is (valid_indices is None)

        if scores is None:
            valid_indices = utils.get_valid_indices(adata)
            representations = torch.tensor(adata.obsm[Keys.REPR][valid_indices])
            scores = self.swav_head.compute_scores(representations)

        leaves_predictions = self._codes_argmax_per_slide(scores, adata, valid_indices)
        leaves_predictions = utils.fill_invalid_indices(leaves_predictions, adata.n_obs, valid_indices)
        adata.obs[Keys.SWAV_CLASSES] = [x if np.isnan(x) else f"N{int(x)}" for x in leaves_predictions]

    def _codes_argmax_per_slide(self, scores: Tensor, adata: AnnData, valid_indices: np.ndarray) -> Tensor:
        slide_ids = adata.obs[Keys.SLIDE_ID].values[valid_indices]

        unique_slide_ids = np.unique(slide_ids)

        ilocs = self.swav_head.get_prototype_ilocs(scores, tissue=adata.uns.get(Keys.UNS_TISSUE, None))

        if len(unique_slide_ids) == 1:
            q = self.swav_head.sinkhorn(scores[:, ilocs])
            return q.argmax(dim=1) if ilocs is Ellipsis else ilocs[q.argmax(dim=1)]

        res = torch.zeros(len(scores), dtype=torch.long)

        for slide_id in unique_slide_ids:
            indices = np.where(slide_ids == slide_id)[0]
            q = self.swav_head.sinkhorn(scores[indices, ilocs])
            res[indices] = q.argmax(dim=1) if ilocs is Ellipsis else ilocs[q.argmax(dim=1)]

        return res

    def _get_adatas(self, adata: AnnData | list[AnnData] | None, slide_key: str | None = None):
        if adata is None:
            return self.adatas
        return utils.prepare_adatas(adata, slide_key=slide_key, var_names=self.cell_embedder.gene_names)[0]

    def plot_niches_hierarchy(
        self,
        max_level: int = 10,
        hline_level: int | list[int] | None = None,
        leaf_font_size: int = 8,
        **kwargs,
    ):
        """Plot the niches hierarchy as a dendogram.

        Args:
            max_level: Maximum level to be plot.
            hline_level: If not `None`, a red line will ne drawn at this/these level(s).
            leaf_font_size: The font size for the leaf labels.
        """
        plot._niches_hierarchy(
            self.swav_head.clustering,
            max_level=max_level,
            hline_level=hline_level,
            leaf_font_size=leaf_font_size,
            **kwargs,
        )

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
    def from_pretrained(self, model_name_or_path: str, **kwargs: int) -> "Novae":
        """Load a pretrained `Novae` model from HuggingFace Hub.

        Args:
            model_name_or_path: Name of the model (or path to the local model).
            **kwargs: Optional kwargs for Hugging Face [`from_pretrained`](https://huggingface.co/docs/huggingface_hub/v0.24.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin.from_pretrained) method.

        Returns:
            A pretrained `Novae` model.
        """
        model = super().from_pretrained(model_name_or_path, **kwargs)
        model.mode.pretrained()
        model._checkpoint = model_name_or_path
        return model

    def save_pretrained(
        self,
        save_directory: str,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """Save a pretrained `Novae` model to a directory.

        Args:
            save_directory: Path to the directory where the model will be saved.
            **kwargs: Do not use. These are used to push a new model on HuggingFace Hub.
        """

        super().save_pretrained(
            save_directory,
            config=dict(self.hparams),
            repo_id=repo_id,
            push_to_hub=push_to_hub,
            **kwargs,
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint[Keys.NOVAE_VERSION] = __version__

    @classmethod
    def load_from_checkpoint(cls, *args, **kwargs) -> "Novae":
        model = super().load_from_checkpoint(*args, **kwargs)
        model.mode.pretrained()
        return model

    @classmethod
    def _load_wandb_artifact(cls, name: str, map_location: str = "cpu", **kwargs: int) -> "Novae":
        artifact_path = utils._load_wandb_artifact(name) / "model.ckpt"

        try:
            model = cls.load_from_checkpoint(artifact_path, map_location=map_location, strict=False, **kwargs)
        except:
            ckpt_version = torch.load(artifact_path, map_location=map_location).get(Keys.NOVAE_VERSION, "unknown")
            raise ValueError(f"The model was trained with `novae=={ckpt_version}`, but your version is {__version__}")

        model._checkpoint = name
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

    def _parse_hardware_args(
        self, accelerator: str, num_workers: int | None, return_device: bool = False
    ) -> torch.device | None:
        if accelerator == "cpu" and num_workers:
            log.warn(
                f"Setting `num_workers != 0` with {accelerator=} can be very slow. Consider using a GPU, or setting `num_workers=0`."
            )

        if num_workers is not None:
            self.num_workers = num_workers

        if return_device:
            return utils.parse_device_args(accelerator)

    @utils.format_docs
    def fine_tune(
        self,
        adata: AnnData | list[AnnData],
        slide_key: str | None = None,
        accelerator: str = "cpu",
        max_epochs: int = 1,
        **fit_kwargs: int,
    ):
        """Fine tune a Novae model.

        Args:
            {adata}
            {slide_key}
            max_epochs: Maximum number of training epochs.
            {accelerator}
            **fit_kwargs: Optional kwargs for the [novae.Novae.fit][] method.
        """
        if not hasattr(self.swav_head, "_kmeans_prototypes"):
            # TODO: no need to compute all repr, make it faster
            # TODO: what happens if run twice?
            self.compute_representation(adata=adata, zero_shot=True)

        self.swav_head._prototypes = self.swav_head._kmeans_prototypes
        del self.swav_head._kmeans_prototypes

        self.mode.fine_tune()

        self.fit(adata=adata, slide_key=slide_key, max_epochs=max_epochs, accelerator=accelerator, **fit_kwargs)

    @utils.format_docs
    def fit(
        self,
        adata: AnnData | list[AnnData] | None = None,
        slide_key: str | None = None,
        max_epochs: int = 20,
        accelerator: str = "cpu",
        lr: float = 1e-3,
        num_workers: int | None = None,
        min_delta: float = 0.1,
        patience: int = 3,
        callbacks: list[Callback] | None = None,
        enable_checkpointing: bool = False,
        logger: Logger | list[Logger] | bool = False,
        **kwargs: int,
    ):
        """Train a Novae model. The training will be stopped by early stopping.

        Args:
            {adata}
            {slide_key}
            max_epochs: Maximum number of training epochs.
            {accelerator}
            lr: Model learning rate.
            num_workers: Number of workers for the dataloader.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement (early stopping).
            patience: Number of epochs with no improvement after which training will be stopped (early stopping).
            callbacks: Optional list of Pytorch lightning callbacks.
            enable_checkpointing: Whether to enable model checkpointing.
            logger: The pytorch lightning logger.
            **kwargs: Optional kwargs for the Pytorch Lightning `Trainer` class.
        """
        self.mode.fit()

        if adata is not None:
            self.adatas, _ = utils.prepare_adatas(adata, slide_key=slide_key, var_names=self.cell_embedder.gene_names)

        ### Misc
        self._lr = lr
        self.swav_head.reset_clustering()  # ensure we don't re-use old clusters
        self._parse_hardware_args(accelerator, num_workers)
        self._datamodule = self._init_datamodule()

        ### Callbacks
        early_stopping = EarlyStopping(
            monitor="train/loss_epoch",
            min_delta=min_delta,
            patience=patience,
            check_on_train_epoch_end=True,
        )
        callbacks = [early_stopping] + (callbacks or [])
        enable_checkpointing = enable_checkpointing or any(isinstance(c, ModelCheckpoint) for c in callbacks)

        ### Training
        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            callbacks=callbacks,
            logger=logger,
            enable_checkpointing=enable_checkpointing,
            **kwargs,
        )
        trainer.fit(self, datamodule=self.datamodule)

        self.mode.trained = True
