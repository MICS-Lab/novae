import logging
from typing import Literal

import lightning as L
import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
from torch import Tensor, optim
from torch_geometric.data import Batch

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

        model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")

        model.compute_representations(adata, zero_shot=True)
        model.assign_domains(adata)
        ```
    """

    def __init__(
        self,
        adata: AnnData | list[AnnData] | None = None,
        embedding_size: int = 100,
        min_prototypes_ratio: float = 0.3,
        n_hops_local: int = 2,
        n_hops_view: int = 2,
        temperature: float = 0.1,
        output_size: int = 64,
        heads: int = 8,
        hidden_size: int = 128,
        num_layers: int = 10,
        batch_size: int = 256,
        num_prototypes: int = 1024,
        panel_subset_size: float = 0.8,
        background_noise_lambda: float = 8.0,
        sensitivity_noise_std: float = 0.05,
        scgpt_model_dir: str | None = None,
        var_names: list[str] | None = None,
    ) -> None:
        """

        Args:
            adata: An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.
            embedding_size: Size of the embeddings of the genes (`E` in the article). Do not use it when loading embeddings from scGPT.
            min_prototypes_ratio: Minimum ratio of prototypes to be used for each slide. Use a low value to get highly slide-specific or condition-specific prototypes.
            n_hops_local: Number of hops between a cell and its neighborhood cells.
            n_hops_view: Number of hops between a cell and the origin of a second graph (or 'view').
            temperature: Temperature used in the cross-entropy loss.
            output_size: Size of the representations, i.e. the encoder outputs (`O` in the article).
            heads: Number of heads for the graph encoder.
            hidden_size: Hidden size for the graph encoder.
            num_layers: Number of layers for the graph encoder.
            batch_size: Mini-batch size.
            num_prototypes: Number of prototypes (`K` in the article).
            panel_subset_size: Ratio of genes kept from the panel during augmentation.
            background_noise_lambda: Parameter of the exponential distribution for the noise augmentation.
            sensitivity_noise_std: Standard deviation for the multiplicative for for the noise augmentation.
            scgpt_model_dir: Path to a directory containing a scGPT checkpoint, i.e. a `vocab.json` and a `best_model.pt` file.
            var_names: Only used when loading a pretrained model. Do not use it yourself.
        """
        super().__init__()
        ### Initialize cell embedder and prepare adata(s) object(s)
        if scgpt_model_dir is None:
            self.adatas, var_names = utils.prepare_adatas(adata, var_names=var_names)
            self.cell_embedder = CellEmbedder(var_names, embedding_size)
            self.cell_embedder.pca_init(self.adatas)
        else:
            self.cell_embedder = CellEmbedder.from_scgpt_embedding(scgpt_model_dir)
            embedding_size = self.cell_embedder.embedding_size
            _scgpt_var_names = self.cell_embedder.gene_names
            self.adatas, var_names = utils.prepare_adatas(adata, var_names=_scgpt_var_names)

        self.save_hyperparameters(ignore=["adata", "scgpt_model_dir"])
        self.mode = utils.Mode()

        ### Initialize modules
        self.encoder = GraphEncoder(embedding_size, hidden_size, num_layers, output_size, heads)
        self.augmentation = GraphAugmentation(panel_subset_size, background_noise_lambda, sensitivity_noise_std)
        self.swav_head = SwavHead(self.mode, output_size, num_prototypes, temperature)

        ### Misc
        self._num_workers = 0
        self._model_name = None
        self._datamodule = None
        self.init_slide_queue(self.adatas, min_prototypes_ratio)

    def init_slide_queue(self, adata: AnnData | list[AnnData] | None, min_prototypes_ratio: float) -> None:
        """
        Initialize the slide-queue for the SwAV head.
        This can be used before training (`fit` or `fine_tune`) when there are potentially slide-specific or condition-specific prototypes.

        Args:
            adata: An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.
            min_prototypes_ratio: Minimum ratio of prototypes to be used for each slide. Use a low value to get highly slide-specific or condition-specific prototypes.
        """
        self.hparams.min_prototypes_ratio = min_prototypes_ratio

        if adata is None or min_prototypes_ratio == 1:
            return

        slide_ids = list(utils.unique_obs(adata, Keys.SLIDE_ID))
        if len(slide_ids) > 1:
            self.swav_head.set_min_prototypes(min_prototypes_ratio)
            self.swav_head.init_queue(slide_ids)

    def __repr__(self) -> str:
        info_dict = {
            "Known genes": self.cell_embedder.voc_size,
            "Parameters": utils.pretty_num_parameters(self),
            "Model name": self._model_name,
        }
        return utils.pretty_model_repr(info_dict)

    def __new__(cls, *args, **kwargs) -> "Novae":
        # trick to enable auto-completion despite PyTorchModelHubMixin inheritance
        return super().__new__(cls, *args, **kwargs)

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

    def _prepare_adatas(self, adata: AnnData | list[AnnData] | None):
        if adata is None:
            return self.adatas
        return utils.prepare_adatas(adata, var_names=self.cell_embedder.gene_names)[0]

    def _init_datamodule(
        self, adata: AnnData | list[AnnData] | None = None, sample_cells: int | None = None, **kwargs: int
    ):
        return NovaeDatamodule(
            self._to_anndata_list(adata),
            cell_embedder=self.cell_embedder,
            batch_size=self.hparams.batch_size,
            n_hops_local=self.hparams.n_hops_local,
            n_hops_view=self.hparams.n_hops_view,
            num_workers=self._num_workers,
            sample_cells=sample_cells,
            **kwargs,
        )

    def configure_optimizers(self):
        lr = self._lr if hasattr(self, "_lr") else 1e-3
        return optim.Adam(self.parameters(), lr=lr)

    def _parse_hardware_args(self, accelerator: str, num_workers: int | None, use_device: bool = False) -> None:
        if accelerator == "cpu" and num_workers:
            log.warning(
                "On CPU, `num_workers != 0` can be very slow. Consider using a GPU, or setting `num_workers=0`."
            )

        if accelerator in ["auto", "cuda", "gpu", "mps"] and not num_workers:
            log.warning("On GPU, consider setting `num_workers` (e.g., 4 or 8) for better performance.")

        if num_workers is not None:
            self.num_workers = num_workers

        if use_device:
            device = utils.parse_device_args(accelerator)
            self.to(device)

    def _embed_pyg_data(self, data: Batch) -> Batch:
        if self.training:
            data = self.augmentation(data)
        return self.cell_embedder(data)

    def forward(self, batch: dict[str, Batch]) -> dict[str, Tensor]:
        return {key: self.encoder(self._embed_pyg_data(data)) for key, data in batch.items()}

    def training_step(self, batch: dict[str, Batch], batch_idx: int):
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
        self.swav_head.prototypes.requires_grad_(after_warm_up or self.mode.pretrained)

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
        """Load a pretrained `Novae` model from HuggingFace Hub.

        !!! info "Available model names"
            See [here](https://huggingface.co/collections/MICS-Lab/novae-669cdf1754729d168a69f6bd) the available Novae model names.

        Args:
            model_name_or_path: Name of the model, e.g. `"MICS-Lab/novae-1-medium"`, or path to the local model.
            **kwargs: Optional kwargs for Hugging Face [`from_pretrained`](https://huggingface.co/docs/huggingface_hub/v0.24.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin.from_pretrained) method.

        Returns:
            A pretrained `Novae` model.
        """
        model = super().from_pretrained(model_name_or_path, **kwargs)

        model.mode.from_pretrained()
        model._model_name = model_name_or_path
        model.cell_embedder.embedding.weight.requires_grad_(False)

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
    def _load_wandb_artifact(cls, model_name: str, map_location: str = "cpu", **kwargs: int) -> "Novae":
        artifact_path = utils.load_wandb_artifact(model_name) / "model.ckpt"

        try:
            model = cls.load_from_checkpoint(artifact_path, map_location=map_location, strict=False, **kwargs)
        except TypeError:
            ckpt_version = torch.load(artifact_path, map_location=map_location).get(Keys.NOVAE_VERSION, "unknown")
            raise ValueError(f"The model was trained with `novae=={ckpt_version}`, but your version is {__version__}")

        model.mode.from_pretrained()
        model._model_name = model_name
        return model

    @torch.no_grad()
    def compute_representations(
        self,
        adata: AnnData | list[AnnData] | None = None,
        *,
        zero_shot: bool = False,
        reference: str | int | Literal["all", "largest"] = "largest",
        accelerator: str = "cpu",
        num_workers: int | None = None,
    ) -> None:
        """Compute the latent representation of Novae for all cells neighborhoods.

        Note:
            Representations are saved in `adata.obsm["novae_latent"]`

        Args:
            adata: An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.
            zero_shot: If `True`, the model will be used in zero-shot mode, i.e. without training. In this case, the model will assign each cell to a leaf based on the latent representations.
            reference: Use only if `zero_shot=True`. Reference slide to use for the new prototypes. Can be the AnnData index, a unique slide id, or one of `["all", "largest"]`.
            accelerator: Accelerator to use. For instance, `'cuda'`, `'cpu'`, or `'auto'`. See Pytorch Lightning for more details.
            num_workers: Number of workers for the dataloader.
        """
        assert self.mode.trained, "Novae must be trained first, so consider running `model.fit()`"

        self.mode.zero_shot = zero_shot
        self.training = False
        self._parse_hardware_args(accelerator, num_workers, use_device=True)

        if adata is None and len(self.adatas) == 1:  # using existing datamodule
            adatas = self.adatas
            self._compute_representations_datamodule(self.adatas[0], self.datamodule)
        else:
            adatas = self._prepare_adatas(adata)
            for adata in adatas:
                datamodule = self._init_datamodule(adata)
                self._compute_representations_datamodule(adata, datamodule)

        if self.mode.zero_shot:
            self.assign_to_kmeans_prototypes(adatas, reference)

    def assign_to_kmeans_prototypes(
        self, adatas: AnnData | list[AnnData], reference: str | int | Literal["all", "largest"]
    ):
        """Compute prototypes based on the latent representations, and assign each cell to a leaf."""
        adatas = [adatas] if isinstance(adatas, AnnData) else adatas

        adatas_refs = _get_reference(adatas, reference)
        adatas_refs = [adatas_refs] if isinstance(adatas_refs, AnnData) else adatas_refs

        latent = np.concatenate([adata.obsm[Keys.REPR][utils.valid_indices(adata)] for adata in adatas_refs])
        self.swav_head.set_kmeans_prototypes(latent)
        self.swav_head.reset_clustering(only_zero_shot=True)

        for adata in adatas:
            self._compute_leaves(adata, None, None)

    @torch.no_grad()
    def _compute_representations_datamodule(
        self, adata: AnnData | None, datamodule: NovaeDatamodule, return_representations: bool = False
    ) -> Tensor | None:
        valid_indices = datamodule.dataset.valid_indices[0]
        representations, projections = [], []

        for batch in utils.tqdm(datamodule.predict_dataloader(), desc="Computing representations"):
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
        adata.obs[Keys.LEAVES] = [x if np.isnan(x) else f"D{int(x)}" for x in leaves_predictions]

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

    def plot_prototype_weights(self, assign_zeros: bool = True, **kwargs: int):
        """Plot the weights of the prototypes per slide."""

        assert (
            self.swav_head.queue is not None
        ), "Swav queue not initialized. Initialize it with `model.init_slide_queue(...)`, then train or fine-tune the model."

        weights, thresholds = self.swav_head.queue_weights()
        weights, thresholds = weights.numpy(force=True), thresholds.numpy(force=True)

        if assign_zeros:
            for i in range(len(weights)):
                below_threshold_ilocs = np.where(weights[i] < thresholds)[0]
                if len(thresholds) - len(below_threshold_ilocs) >= self.swav_head.min_prototypes:
                    weights[i, below_threshold_ilocs] = 0
                else:
                    n_missing = len(thresholds) - self.swav_head.min_prototypes
                    ilocs = below_threshold_ilocs[
                        np.argpartition(weights[i, below_threshold_ilocs], n_missing)[:n_missing]
                    ]
                    weights[i, ilocs] = 0

        plot._weights_clustermap(weights, self.adatas, list(self.swav_head.slide_label_encoder.keys()), **kwargs)

    def plot_prototype_covariance(self, vmax: float | None = None, **kwargs):
        covariance = np.cov(self.swav_head.prototypes.data.numpy(force=True))

        vmax = vmax or covariance.max()

        plot._weights_clustermap(
            covariance, None, [], show_yticklabels=False, show_tissue_legend=False, vmax=vmax, **kwargs
        )

    def assign_domains(
        self,
        adata: AnnData | list[AnnData] | None = None,
        level: int | None = 7,
        n_domains: int | None = None,
        resolution: float | None = None,
        key_added: str | None = None,
    ) -> str:
        """Assign a domain to each cell based on the "leaves" classes.
        It either (i) uses a specific `level` of the hierarchical tree,
        (ii) enforces a precise number of `n_domains`,
        or (iii) uses the Leiden clustering with a specific `resolution`.

        Note:
            You'll need to run [novae.Novae.compute_representations][] first.

            The domains are saved in `adata.obs["novae_domains_X]`, where `X` is the `level` argument or `res{resolution}` if using Leiden.

        Args:
            adata: An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.
            level: Level of the domains hierarchical tree (i.e., number of different domains to assigned).
            n_domains: If `level` is not providing the desired number of domains, use this argument to enforce a precise number of domains.
            resolution: Resolution for the Leiden clustering. If `None`, uses the hierarchical clustering instead.
            key_added: The spatial domains will be saved in `adata.obs[key_added]`. By default, it is `adata.obs["novae_domains_X]`, where `X` is the `level` argument.

        Returns:
            The name of the key added to `adata.obs`.
        """
        adatas = self._to_anndata_list(adata)

        assert all(
            Keys.LEAVES in adata.obs for adata in adatas
        ), f"Did not found `adata.obs['{Keys.LEAVES}']`. Please run `model.compute_representations(...)` first"

        if resolution is not None:
            _leiden_codes = self._leiden_prototypes(resolution=resolution)

            key_added = f"{Keys.DOMAINS_PREFIX}res{resolution}"

            for adata in adatas:
                adata.obs[key_added] = adata.obs[Keys.LEAVES].map(
                    lambda x: f"L{_leiden_codes[int(x[1:])]}" if isinstance(x, str) else x
                )

            return key_added

        if n_domains is not None:
            leaves_indices = utils.unique_leaves_indices(adatas)
            level = self.swav_head.find_level(leaves_indices, n_domains)
            return self.assign_domains(adatas, level=level, key_added=key_added)

        key_added = f"{Keys.DOMAINS_PREFIX}{level}" if key_added is None else key_added

        for adata in adatas:
            adata.obs[key_added] = self.swav_head.map_leaves_domains(adata.obs[Keys.LEAVES], level)
            adata.obs[key_added] = adata.obs[key_added].astype("category")

        return key_added

    @torch.no_grad()
    def _leiden_prototypes(self, resolution: float = 1, return_codes: bool = True) -> AnnData | np.ndarray:
        adata_proto = AnnData(self.swav_head.prototypes.numpy(force=True))

        sc.pp.pca(adata_proto)
        sc.pp.neighbors(adata_proto)
        sc.tl.leiden(adata_proto, flavor="igraph", resolution=resolution)

        if return_codes:
            return adata_proto.obs["leiden"].values.codes
        return adata_proto

    def batch_effect_correction(self, adata: AnnData | list[AnnData] | None = None, obs_key: str | None = None):
        """Correct batch effects from the spatial representations of cells.

        !!! info
            The corrected spatial representations will be saved in `adata.obsm["novae_latent_corrected"]`.

        Args:
            adata: An `AnnData` object, or a list of `AnnData` objects.
            obs_key: Optional key in `adata.obs` containing the domains to use for batch correction. If not provided, the key will be automatically selected.
        """
        adatas = self._to_anndata_list(adata)
        obs_key = utils.check_available_domains_key(adatas, obs_key)

        utils.batch_effect_correction(adatas, obs_key)

    def fine_tune(
        self,
        adata: AnnData | list[AnnData],
        *,
        reference: str | int | Literal["all", "largest"] = "largest",
        accelerator: str = "cpu",
        num_workers: int | None = None,
        min_prototypes_ratio: float = 0.3,
        lr: float = 1e-3,
        max_epochs: int = 4,
        **fit_kwargs: int,
    ):
        """Fine tune a pretrained Novae model. This will update the prototypes with the new data, and `fit` for one or a few epochs.

        Args:
            adata: An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.
            reference: Reference slide to use for the new prototypes. Can be the AnnData index, a unique slide id, or one of `["all", "largest"]`.
            accelerator: Accelerator to use. For instance, `'cuda'`, `'cpu'`, or `'auto'`. See Pytorch Lightning for more details.
            num_workers: Number of workers for the dataloader.
            min_prototypes_ratio: Minimum ratio of prototypes to be used for each slide. Use a low value to get highly slide-specific or condition-specific prototypes.
            lr: Model learning rate.
            max_epochs: Maximum number of training epochs.
            **fit_kwargs: Optional kwargs for the [novae.Novae.fit][] method.
        """
        self.mode.fine_tune()
        self._parse_hardware_args(accelerator, num_workers, use_device=True)

        assert adata is not None, "Please provide an AnnData object to fine-tune the model."

        datamodule = self._init_datamodule(
            self._prepare_adatas(_get_reference(adata, reference)), sample_cells=Nums.DEFAULT_SAMPLE_CELLS
        )
        latent = self._compute_representations_datamodule(None, datamodule, return_representations=True)
        self.swav_head.set_kmeans_prototypes(latent.numpy(force=True))

        self.swav_head._prototypes = self.swav_head._kmeans_prototypes
        del self.swav_head._kmeans_prototypes

        self.init_slide_queue(adata, min_prototypes_ratio)

        self.fit(
            adata=adata,
            max_epochs=max_epochs,
            accelerator=accelerator,
            num_workers=num_workers,
            lr=lr,
            **fit_kwargs,
        )

    def fit(
        self,
        adata: AnnData | list[AnnData] | None = None,
        max_epochs: int = 20,
        accelerator: str = "cpu",
        num_workers: int | None = None,
        lr: float = 1e-3,
        min_delta: float = 0.1,
        patience: int = 3,
        callbacks: list[Callback] | None = None,
        logger: Logger | list[Logger] | bool = False,
        **trainer_kwargs: int,
    ):
        """Train a Novae model. The training will be stopped by early stopping.

        !!! warn
            If you loaded a pretrained model, use [novae.Novae.fine_tune][] instead.

        Args:
            adata: An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.
            max_epochs: Maximum number of training epochs.
            accelerator: Accelerator to use. For instance, `'cuda'`, `'cpu'`, or `'auto'`. See Pytorch Lightning for more details.
            num_workers: Number of workers for the dataloader.
            lr: Model learning rate.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement (early stopping).
            patience: Number of epochs with no improvement after which training will be stopped (early stopping).
            callbacks: Optional list of Pytorch lightning callbacks.
            logger: The pytorch lightning logger.
            **trainer_kwargs: Optional kwargs for the Pytorch Lightning `Trainer` class.
        """
        self.mode.fit()

        if adata is not None:
            self.adatas, _ = utils.prepare_adatas(adata, var_names=self.cell_embedder.gene_names)

        ### Misc
        self._lr = lr
        self.swav_head.reset_clustering()  # ensure we don't re-use old clusters
        self._parse_hardware_args(accelerator, num_workers)
        self._datamodule = self._init_datamodule()

        _train(
            self,
            self.datamodule,
            accelerator,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            callbacks=callbacks,
            logger=logger,
            **trainer_kwargs,
        )

        self.mode.trained = True


def _train(
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    accelerator: str,
    max_epochs: int = 50,
    patience: int = 3,
    min_delta: float = 0,
    callbacks: list[Callback] | None = None,
    logger: Logger | list[Logger] | bool = False,
    **trainer_kwargs: int,
):
    """Internal function to train a LightningModule with early stopping."""

    early_stopping = EarlyStopping(
        monitor="train/loss_epoch",
        min_delta=min_delta,
        patience=patience,
        check_on_train_epoch_end=True,
    )
    callbacks = [early_stopping] + (callbacks or [])
    enable_checkpointing = any(isinstance(c, ModelCheckpoint) for c in callbacks)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        **trainer_kwargs,
    )
    trainer.fit(model, datamodule=datamodule)


def _get_reference(
    adata: AnnData | list[AnnData], reference: str | int | Literal["all", "largest"]
) -> AnnData | list[AnnData]:
    if reference == "all":
        return adata

    if isinstance(reference, int):
        assert isinstance(adata, list), "When providing an index, you must provide a list of AnnData objects."
        return adata[reference]

    if reference == "largest":

        def _select_largest_slide(adata: AnnData):
            counts = adata.obs[Keys.SLIDE_ID].value_counts()
            return counts.max(), adata[adata.obs[Keys.SLIDE_ID] == counts.idxmax()]

        if isinstance(adata, AnnData):
            return _select_largest_slide(adata)[1]
        else:
            return max([_select_largest_slide(_adata) for _adata in adata])[1]

    assert isinstance(reference, str), f"Invalid type for `reference`: {type(reference)}"

    adatas = [adata] if isinstance(adata, AnnData) else adata
    for adata in adatas:
        if reference in adata.obs[Keys.SLIDE_ID].cat.categories:
            return adata[adata.obs[Keys.SLIDE_ID] == reference]

    raise ValueError(f"Did not found slide id `{reference}` in the provided AnnData object(s).")
