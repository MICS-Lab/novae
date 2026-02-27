"""
Novae model training with Weight & Biases monitoring
This is **not** the actual Novae source code. Instead, see the `novae` directory
"""

# isort: skip_file
# else segmentation fault on our-HPC-specific env
import pyarrow  # noqa: F401

import argparse

from anndata import AnnData

import novae
from novae._constants import Keys
from novae.data._load import load_local_dataset

from .utils import get_callbacks, init_wandb_logger, post_training, read_config


def main(args: argparse.Namespace) -> None:
    config = read_config(args)

    adatas = load_local_dataset(config.data.train_dataset, files_black_list=config.data.files_black_list)
    adatas_val = load_local_dataset(config.data.val_dataset) if config.data.val_dataset else None

    _check_sid(adatas)
    if adatas_val is not None:
        _check_sid(adatas_val)

    logger = init_wandb_logger(config)
    callbacks = get_callbacks(config, adatas_val)

    if config.wandb_artefact is not None:
        model = novae.Novae._load_wandb_artifact(config.wandb_artefact)

        if not config.zero_shot:
            model.fine_tune(adatas, logger=logger, callbacks=callbacks, **config.fit_kwargs)
    else:
        model = novae.Novae(adatas, **config.model_kwargs)
        model.fit(logger=logger, callbacks=callbacks, **config.fit_kwargs)

    post_training(model, adatas, config)


def _check_sid(adatas: list[AnnData]):
    for adata in adatas:
        if Keys.SLIDE_ID not in adata.obs:
            assert "slide_id" in adata.obs
            adata.obs[Keys.SLIDE_ID] = adata.obs["slide_id"].astype("category")  # backwards compatibility


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Fullname of the YAML config to be used for training (see under the `config` directory)",
    )
    parser.add_argument(
        "-s",
        "--sweep",
        nargs="?",
        default=False,
        const=True,
        help="Whether it is a sweep or not",
    )

    main(parser.parse_args())
