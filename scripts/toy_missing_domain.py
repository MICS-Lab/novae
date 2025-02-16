import argparse

import wandb
from sklearn.metrics.cluster import adjusted_rand_score

import novae
from novae.monitor.log import log_plt_figure

from .utils import init_wandb_logger, read_config


def main(args: argparse.Namespace) -> None:
    config = read_config(args)

    adatas = novae.utils.toy_dataset(n_panels=2, xmax=2_000, n_domains=7, n_vars=300)

    adatas[1] = adatas[1][adatas[1].obs["domain"] != "domain_6", :].copy()
    novae.spatial_neighbors(adatas)

    logger = init_wandb_logger(config)

    model = novae.Novae(adatas, **config.model_kwargs)
    model.fit(logger=logger, **config.fit_kwargs)

    model.compute_representations(adatas)
    obs_key = model.assign_domains(adatas, level=7)

    novae.plot.domains(adatas, obs_key=obs_key, show=False)
    log_plt_figure(f"domains_{obs_key}")

    ari = adjusted_rand_score(adatas[0].obs[obs_key], adatas[0].obs["domain"])

    adatas[0] = adatas[0][adatas[0].obs["domain"] != "domain_6", :].copy()
    accuracy = (adatas[0].obs[obs_key].values.astype(str) == adatas[1].obs["domain"].values.astype(str)).mean()

    wandb.log({"metrics/score": ari * accuracy, "metrics/ari": ari, "metrics/accuracy": accuracy})


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
