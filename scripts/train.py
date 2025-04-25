"""
Novae model training with Weight & Biases monitoring
This is **not** the actual Novae source code. Instead, see the `novae` directory
"""

import argparse

import novae

from .utils import get_callbacks, init_wandb_logger, post_training, read_config


def main(args: argparse.Namespace) -> None:
    config = read_config(args)

    adatas = novae.data.load.load_local_dataset(
        config.data.train_dataset, files_black_list=config.data.files_black_list
    )
    adatas_val = novae.data.load.load_local_dataset(config.data.val_dataset) if config.data.val_dataset else None

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
