import importlib
import logging
from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.trainer.connectors.accelerator_connector import _AcceleratorConnector

from .._constants import Keys

log = logging.getLogger(__name__)


def unique_obs(adata: AnnData | list[AnnData], obs_key: str) -> set:
    if isinstance(adata, list):
        return set.union(*[unique_obs(adata_, obs_key) for adata_ in adata])
    return set(adata.obs[obs_key].dropna().unique())


def unique_leaves_indices(adata: AnnData | list[AnnData]) -> set:
    leaves = unique_obs(adata, Keys.LEAVES)
    return np.array([int(x[1:]) for x in leaves])


def valid_indices(adata: AnnData) -> np.ndarray:
    return np.where(adata.obs[Keys.IS_VALID_OBS])[0]


def lower_var_names(var_names: pd.Index | list[str]) -> pd.Index | list[str]:
    if isinstance(var_names, pd.Index):
        return var_names.str.lower()
    return [name.lower() for name in var_names]


def fill_invalid_indices(
    out: np.ndarray,
    n_obs: int,
    valid_indices: list[int],
    fill_value: float | str = np.nan,
    dtype: object = None,
) -> np.ndarray:
    dtype = np.float32 if dtype is None else dtype

    if isinstance(fill_value, str):
        dtype = object

    res = np.full((n_obs, *out.shape[1:]), fill_value, dtype=dtype)
    res[valid_indices] = out
    return res


def parse_device_args(accelerator: str = "cpu") -> torch.device:
    """Updated from scvi-tools"""
    connector = _AcceleratorConnector(accelerator=accelerator)
    _accelerator = connector._accelerator_flag
    _devices = connector._devices_flag

    if _accelerator == "cpu":
        return torch.device("cpu")

    if isinstance(_devices, list):
        device_idx = _devices[0]
    elif isinstance(_devices, str) and "," in _devices:
        device_idx = _devices.split(",")[0]
    else:
        device_idx = _devices

    return torch.device(f"{_accelerator}:{device_idx}")


def repository_root() -> Path:
    """Get the path to the root of the repository (dev-mode users only)

    Returns:
        `novae` repository path
    """
    path = Path(__file__).parents[2]

    if path.name != "novae":
        log.warning(f"Trying to get the novae repository path, but it seems it was not installed in dev mode: {path}")

    return path


def wandb_log_dir() -> Path:
    return repository_root() / "wandb"


def tqdm(*args, desc="DataLoader", **kwargs):
    # check if ipywidgets is installed before importing tqdm.auto
    # to ensure it won't fail and a progress bar is displayed
    if importlib.util.find_spec("ipywidgets") is not None:
        from tqdm.auto import tqdm as _tqdm
    else:
        from tqdm import tqdm as _tqdm

    return _tqdm(*args, desc=desc, **kwargs)


def pretty_num_parameters(model: torch.nn.Module) -> str:
    n_params = sum(p.numel() for p in model.parameters())

    if n_params < 1_000_000:
        return f"{n_params / 1_000:.1f}K"

    return f"{n_params / 1_000_000:.1f}M"


def pretty_model_repr(info_dict: dict[str, str], model_name: str = "Novae") -> str:
    rows = [f"{model_name} model"] + [f"{k}: {v}" for k, v in info_dict.items()]
    return "\n   ├── ".join(rows[:-1]) + "\n   └── " + rows[-1]


def iter_slides(adatas: AnnData | list[AnnData]):
    """Iterate over all slides.

    Args:
        adatas: One or a list of AnnData object(s).

    Yields:
        One `AnnData` per slide.
    """
    if isinstance(adatas, AnnData):
        adatas = [adatas]

    for adata in adatas:
        slide_ids = adata.obs[Keys.SLIDE_ID].unique()

        if len(slide_ids) == 1:
            yield adata
            continue

        for slide_id in slide_ids:
            yield adata[adata.obs[Keys.SLIDE_ID] == slide_id]


def train(
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


def get_reference(
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
