import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from lightning import Trainer
from lightning.pytorch.callbacks import Callback

from .._constants import Keys
from ..model import Novae
from .eval import heuristic, mean_fide_score
from .log import log_plt_figure, save_pdf_figure


class LogProtoCovCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, model: Novae) -> None:
        C = model.swav_head.prototypes.data.numpy(force=True)

        plt.figure(figsize=(10, 10))
        sns.clustermap(np.cov(C))
        log_plt_figure("prototypes_covariance")


class LogTissuePrototypeWeights(Callback):
    def on_train_epoch_end(self, trainer: Trainer, model: Novae) -> None:
        if model.swav_head.queue is None:
            return

        model.plot_prototype_weights()
        save_pdf_figure(f"tissue_prototype_weights_e{model.current_epoch}")
        log_plt_figure("tissue_prototype_weights")


class ValidationCallback(Callback):
    def __init__(
        self,
        adatas: list[AnnData] | None,
        accelerator: str = "cpu",
        num_workers: int = 0,
        slide_name_key: str = "slide_id",
        k: int = 7,
        res: float = 0.5,
    ):
        assert adatas is None or len(adatas) == 1, "ValidationCallback only supports single slide mode for now"
        self.adata = adatas[0] if adatas is not None else None
        self.accelerator = accelerator
        self.num_workers = num_workers
        self.slide_name_key = slide_name_key
        self.k = k
        self.res = res

        self._max_heuristic = 0.0

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        if self.adata is None:
            return

        model.mode.trained = True  # trick to avoid assert error in compute_representations

        model.compute_representations(
            self.adata, accelerator=self.accelerator, num_workers=self.num_workers, zero_shot=True
        )
        model.swav_head.hierarchical_clustering()

        obs_key = model.assign_domains(self.adata, n_domains=self.k)

        plt.figure()
        sc.pl.spatial(self.adata, color=obs_key, spot_size=20, img_key=None, show=False)
        slide_name_key = self.slide_name_key if self.slide_name_key in self.adata.obs else Keys.SLIDE_ID
        log_plt_figure(f"val_{self.k}_{self.adata.obs[slide_name_key].iloc[0]}")

        fide = mean_fide_score(self.adata, obs_key=obs_key, n_classes=self.k)
        model.log("metrics/val_mean_fide_score", fide)

        heuristic_ = heuristic(self.adata, obs_key=obs_key, n_classes=self.k)
        model.log("metrics/val_heuristic", heuristic_)

        self._max_heuristic = max(self._max_heuristic, heuristic_)
        model.log("metrics/val_max_heuristic", self._max_heuristic)

        obs_key = model.assign_domains(self.adata, resolution=self.res)

        plt.figure()
        sc.pl.spatial(self.adata, color=obs_key, spot_size=20, img_key=None, show=False)
        log_plt_figure(f"val_res{self.res}_{self.adata.obs[slide_name_key].iloc[0]}")

        fide = mean_fide_score(self.adata, obs_key=obs_key)
        model.log(f"metrics/val_mean_fide_score_res{self.res}", fide)

        n_classes = len(self.adata.obs[obs_key].cat.categories)
        heuristic_ = heuristic(self.adata, obs_key=obs_key, n_classes=n_classes)
        model.log(f"metrics/val_heuristic_res{self.res}", heuristic_)

        model.mode.zero_shot = False


class PrototypeUMAPCallback(Callback):
    MAX_SLIDES: int = 8
    LEVEL: int = 15

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        adata_proto = AnnData(model.swav_head.prototypes.data.numpy(force=True))

        sc.pp.neighbors(adata_proto)
        sc.tl.umap(adata_proto)

        obs_key = f"level_{self.LEVEL}"

        adata_proto.obs[obs_key] = model.swav_head.clusters_levels[-self.LEVEL]
        adata_proto.obs[obs_key] = adata_proto.obs[obs_key].astype("category")

        sc.pl.umap(adata_proto, color=obs_key, show=False)
        log_plt_figure("prototype_umap")

        model.swav_head.reset_clustering()

        weights, _ = model.swav_head.queue_weights()
        weights = weights.numpy(force=True)

        slide_ids = list(model.swav_head.slide_label_encoder.keys())[: self.MAX_SLIDES]
        weights = weights[[model.swav_head.slide_label_encoder[slide_id] for slide_id in slide_ids], :]

        adata_proto.obs[slide_ids] = weights.T

        sc.pl.umap(adata_proto, color=slide_ids, vmax="p95", show=False, ncols=4)
        log_plt_figure("prototype_umap_slide_weights")
