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
        model.compute_representations()
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
        embedding_size: int = 100,
        output_size: int = 64,
        n_hops_local: int = 2,
        n_hops_view: int = 2,
        heads: int = 4,
        hidden_size: int = 64,
        num_layers: int = 10,
        batch_size: int = 512,
        temperature: float = 0.1,
        temperature_weight_proto: float = 0.1,
        num_prototypes: int = 256,
        panel_subset_size: float = 0.6,
        background_noise_lambda: float = 8.0,
        sensitivity_noise_std: float = 0.05,
        min_prototypes_ratio: float = 0.8,
    ) -> None:
        """

        Args:
            {adata}
            {slide_key}
            {scgpt_model_dir}
            {var_names}
            {embedding_size} Do not use it when loading embeddings from scGPT.
            {output_size}
            {n_hops_local}
            {n_hops_view}
            heads: Number of heads for the graph encoder.
            hidden_size: Hidden size for the graph encoder.
            num_layers: Number of layers for the graph encoder.
            batch_size: Mini-batch size.
            {temperature}
            {num_prototypes}
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
            self.mode, output_size, num_prototypes, temperature, temperature_weight_proto, min_prototypes_ratio
        )

        ### Misc
        self._num_workers = 0
        self._checkpoint = None
        self._datamodule = None
        self.init_slide_queue(self.adatas)

    def init_slide_queue(self, adata: AnnData | list[AnnData] | None) -> None:
        if adata is None:
            return

        slide_ids = list(utils.unique_obs(adata, Keys.SLIDE_ID))
        if len(slide_ids) > 1:
            self.mode.queue_mode = True
            self.swav_head.init_queue(slide_ids)

    def __repr__(self) -> str:
        info_dict = {
            "known genes": self.cell_embedder.voc_size,
            "parameters": utils.pretty_num_parameters(self),
            "checkpoint": self._checkpoint,
        }
        return utils.pretty_model_repr(info_dict)

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

    def _to_anndata_list(self, adata: AnnData | list[AnnData] | None) -> list[AnnData]:
        if adata is None:
            assert self.adatas is not None, "No AnnData object found. Please provide an AnnData object."
            return self.adatas
        elif isinstance(adata, AnnData):
            return [adata]
        elif isinstance(adata, list):
            return adata
        else:
            raise ValueError(f"Invalid type for `adata`: {type(adata)}")

    def _prepare_adatas(self, adata: AnnData | list[AnnData] | None, slide_key: str | None = None):
        if adata is None:
            return self.adatas
        return utils.prepare_adatas(adata, slide_key=slide_key, var_names=self.cell_embedder.gene_names)[0]

    def _init_datamodule(self, adata: AnnData | list[AnnData] | None = None, sample_cells: int | None = None):
        return NovaeDatamodule(
            self._to_anndata_list(adata),
            cell_embedder=self.cell_embedder,
            batch_size=self.hparams.batch_size,
            n_hops_local=self.hparams.n_hops_local,
            n_hops_view=self.hparams.n_hops_view,
            num_workers=self._num_workers,
            sample_cells=sample_cells,
        )

    def configure_optimizers(self):
        lr = self._lr if hasattr(self, "_lr") else 1e-3
        return optim.Adam(self.parameters(), lr=lr)

    def _parse_hardware_args(self, accelerator: str, num_workers: int | None, use_device: bool = False) -> None:
        if accelerator == "cpu" and num_workers:
            log.warn("On CPU, `num_workers != 0` can be very slow. Consider using a GPU, or setting `num_workers=0`.")

        if num_workers is not None:
            self.num_workers = num_workers

        if use_device:
            device = utils.parse_device_args(accelerator)
            self.to(device)

    def _embed_pyg_data(self, data: Data) -> Data:
        if self.training:
            data = self.augmentation(data)
        return self.cell_embedder(data)

    def forward(self, batch: dict[str, Data]) -> dict[str, Tensor]:
        return {key: self.encoder(self._embed_pyg_data(data)) for key, data in batch.items()}

    def training_step(self, batch: dict[str, Data], batch_idx: int):
        z_dict: dict[str, Tensor] = self(batch)
        slide_id = batch["main"].get("slide_id", [None])[0]

        loss, mean_entropy_normalized = self.swav_head(z_dict["main"], z_dict["view"], slide_id)

        self._log_progress_bar("loss", loss)
        self._log_progress_bar("entropy", mean_entropy_normalized, on_epoch=False)

        return loss

    def on_train_epoch_start(self):
        self.training = True

        self.datamodule.dataset.shuffle_obs_ilocs()

        after_warm_up = self.current_epoch >= Nums.WARMUP_EPOCHS

        self.swav_head.prototypes.requires_grad_(after_warm_up or not self.mode.freeze_mode)
        self.mode.use_queue = after_warm_up and self.mode.queue_mode

    def _log_progress_bar(self, name: str, value: float, on_epoch: bool = True, **kwargs):
        self.log(
            f"train/{name}",
            value,
            on_epoch=on_epoch,
            on_step=True,
            batch_size=self.hparams.batch_size,
            prog_bar=True,
            **kwargs,
        )

    @classmethod
    def from_pretrained(self, model_name_or_path: str, **kwargs: int) -> "Novae":
        """Load a pretrained `Novae` model from HuggingFace Hub. See [here](https://huggingface.co/collections/MICS-Lab/novae-669cdf1754729d168a69f6bd) the available Novae model names.

        Args:
            model_name_or_path: Name of the model, e.g. `MICS-Lab/novae-example-v0`, or path to the local model.
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
        **kwargs: int,
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
        artifact_path = utils.load_wandb_artifact(name) / "model.ckpt"

        try:
            model = cls.load_from_checkpoint(artifact_path, map_location=map_location, strict=False, **kwargs)
        except:
            ckpt_version = torch.load(artifact_path, map_location=map_location).get(Keys.NOVAE_VERSION, "unknown")
            raise ValueError(f"The model was trained with `novae=={ckpt_version}`, but your version is {__version__}")

        model._checkpoint = name
        return model

    @utils.format_docs
    @utils.requires_fit
    @torch.no_grad()
    def compute_representations(
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
        self._parse_hardware_args(accelerator, num_workers, use_device=True)

        if adata is None and len(self.adatas) == 1:  # using existing datamodule
            adatas = self.adatas
            self._compute_representations_datamodule(self.adatas[0], self.datamodule)
        else:
            adatas = self._prepare_adatas(adata, slide_key=slide_key)
            for adata in adatas:
                datamodule = self._init_datamodule(adata)
                self._compute_representations_datamodule(adata, datamodule)

        if self.mode.zero_shot:
            latent = np.concatenate([adata.obsm[Keys.REPR][utils.valid_indices(adata)] for adata in adatas])
            self.swav_head.set_kmeans_prototypes(latent)

            for adata in adatas:
                self._compute_leaves(adata, None, None)

    @torch.no_grad()
    def _compute_representations_datamodule(
        self, adata: AnnData | None, datamodule: NovaeDatamodule, return_representations: bool = False
    ) -> Tensor | None:
        valid_indices = datamodule.dataset.valid_indices[0]
        representations, projections = [], []

        for batch in utils.tqdm(datamodule.predict_dataloader()):
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            batch_repr = self.encoder(self._embed_pyg_data(batch["main"]))

            representations.append(batch_repr)

            if not self.mode.zero_shot:
                batch_projections = self.swav_head.projection(batch_repr)
                projections.append(batch_projections)

        representations = torch.cat(representations)

        if return_representations:
            return representations

        adata.obsm[Keys.REPR] = utils.fill_invalid_indices(representations, adata.n_obs, valid_indices, fill_value=0)

        if not self.mode.zero_shot:
            projections = torch.cat(projections)
            self._compute_leaves(adata, projections, valid_indices)

    def _compute_leaves(self, adata: AnnData, projections: Tensor | None, valid_indices: np.ndarray | None):
        assert (projections is None) is (valid_indices is None)

        if projections is None:
            valid_indices = utils.valid_indices(adata)
            representations = torch.tensor(adata.obsm[Keys.REPR][valid_indices])
            projections = self.swav_head.projection(representations)

        leaves_predictions = projections.argmax(dim=1).numpy(force=True)
        leaves_predictions = utils.fill_invalid_indices(leaves_predictions, adata.n_obs, valid_indices)
        adata.obs[Keys.LEAVES] = [x if np.isnan(x) else f"N{int(x)}" for x in leaves_predictions]

    def plot_domains_hierarchy(
        self,
        max_level: int = 10,
        hline_level: int | list[int] | None = None,
        leaf_font_size: int = 8,
        **kwargs,
    ):
        """Plot the domains hierarchy as a dendogram.

        Args:
            max_level: Maximum level to be plot.
            hline_level: If not `None`, a red line will ne drawn at this/these level(s).
            leaf_font_size: The font size for the leaf labels.
        """
        plot._domains_hierarchy(
            self.swav_head.clustering,
            max_level=max_level,
            hline_level=hline_level,
            leaf_font_size=leaf_font_size,
            **kwargs,
        )

    def plot_prototype_weights(self, **kwargs: int):
        weights = self.swav_head.queue_weights().numpy(force=True)

        where_enough_prototypes = (weights >= Nums.QUEUE_WEIGHT_THRESHOLD).sum(1) >= self.swav_head.min_prototypes
        for i in np.where(where_enough_prototypes)[0]:
            weights[i, weights[i] < Nums.QUEUE_WEIGHT_THRESHOLD] = 0
        for i in np.where(~where_enough_prototypes)[0]:
            indices_0 = np.argsort(weights[i])[: -self.swav_head.min_prototypes]
            weights[i, indices_0] = 0

        plot._weights_clustermap(weights, self.adatas, list(self.swav_head.slide_label_encoder.keys()), **kwargs)

    @utils.format_docs
    def assign_domains(
        self,
        adata: AnnData | list[AnnData] | None = None,
        level: int = 10,
        n_domains: int | None = None,
        key_added: str | None = None,
    ) -> str:
        """Assign a domain to each cell based on the "leaves" classes.

        Note:
            You'll need to run `model.compute_representations(...)` first.

            The domains are saved in `adata.obs["novae_domains_X]`, where `X` is the `level` argument.

        Args:
            {adata}
            level: Level of the domains hierarchical tree (i.e., number of different domains to assigned).
            n_domains: If `level` is not providing the desired number of domains, use this argument to enforce a precise number of domains.
            key_added: The spatial domains will be saved in `adata.obs[key_added]`. By default, it is `adata.obs["novae_domains_X]`, where `X` is the `level` argument.

        Returns:
            The name of the key added to `adata.obs`.
        """
        adatas = self._to_anndata_list(adata)

        assert all(
            Keys.LEAVES in adata.obs for adata in adatas
        ), "Did not found `adata.obs['leaves']`. Please run `model.compute_representations(...)` first"

        if n_domains is not None:
            leaves_indices = utils.unique_leaves_indices(adatas)
            level = self.swav_head.find_level(leaves_indices, n_domains)
            return self.assign_domains(adatas, level=level, key_added=key_added)

        key_added = f"{Keys.DOMAINS_PREFIX}{level}" if key_added is None else key_added

        for adata in adatas:
            adata.obs[key_added] = self.swav_head.map_leaves_domains(adata.obs[Keys.LEAVES], level)

        return key_added

    def batch_effect_correction(self, adata: AnnData | list[AnnData] | None = None, obs_key: str | None = None):
        adatas = self._to_anndata_list(adata)
        obs_key = utils.check_available_domains_key(adatas, obs_key)

        utils.batch_effect_correction(adatas, obs_key)

    @utils.format_docs
    def fine_tune(
        self,
        adata: AnnData | list[AnnData],
        slide_key: str | None = None,
        accelerator: str = "cpu",
        num_workers: int | None = None,
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
        self.mode.fine_tune()
        self._parse_hardware_args(accelerator, num_workers, use_device=True)

        assert adata is not None, "Please provide an AnnData object to fine-tune the model."

        datamodule = self._init_datamodule(self._prepare_adatas(adata), sample_cells=Nums.DEFAULT_SAMPLE_CELLS)
        latent = self._compute_representations_datamodule(None, datamodule, return_representations=True)
        self.swav_head.set_kmeans_prototypes(latent.numpy(force=True))

        self.swav_head._prototypes = self.swav_head._kmeans_prototypes
        del self.swav_head._kmeans_prototypes

        self.fit(
            adata=adata,
            slide_key=slide_key,
            max_epochs=max_epochs,
            accelerator=accelerator,
            num_workers=num_workers,
            **fit_kwargs,
        )

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
        **trainer_kwargs: int,
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
            **trainer_kwargs: Optional kwargs for the Pytorch Lightning `Trainer` class.
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
            **trainer_kwargs,
        )
        trainer.fit(self, datamodule=self.datamodule)

        self.mode.trained = True
