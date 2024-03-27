from __future__ import annotations

from lightning import Trainer
from lightning.pytorch.callbacks import Callback

from ..model import Novae
from .log import log_domains_plots, log_metrics


class ComputeSwavOutputsCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, model: Novae) -> None:
        model.swav_classes()
        model.swav_head.hierarchical_clustering()


class LogDomainsCallback(Callback):
    def __init__(self, **plot_kwargs) -> None:
        super().__init__()
        self.plot_kwargs = plot_kwargs

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        log_domains_plots(model, model.adatas, **self.plot_kwargs)


class EvalCallback(Callback):
    def __init__(self, **metrics_kwargs) -> None:
        super().__init__()
        self.metrics_kwargs = metrics_kwargs

    def on_train_epoch_end(self, trainer: Trainer, model: Novae):
        log_metrics(model.adatas, **self.metrics_kwargs)
