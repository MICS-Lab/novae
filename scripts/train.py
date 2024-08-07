"""
Novae model training with Weight & Biases monitoring
This is **not** the actual Novae source code. Instead, see the `novae` directory
"""

from __future__ import annotations

import argparse

import novae

from .utils import get_callbacks, init_wandb_logger, post_training, read_config


def main(args: argparse.Namespace) -> None:
    config = read_config(args)

    adatas = novae.utils._load_dataset(config["data"]["train_dataset"])

    _val_dataset = config["data"].get("val_dataset")
    adatas_val = novae.utils._load_dataset(_val_dataset) if _val_dataset else None

    logger = init_wandb_logger(config)
    callbacks = get_callbacks(config, adatas_val)

    if config.get("wandb_artifact"):
        model = novae.Novae._load_wandb_artifact(config["wandb_artifact"])
        model.fine_tune(adatas, logger=logger, callbacks=callbacks, **config["fit_kwargs"])
    else:
        model = novae.Novae(adatas, **config["model_kwargs"])
        model.fit(logger=logger, callbacks=callbacks, **config["fit_kwargs"])

    post_training(model, adatas, config)


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
