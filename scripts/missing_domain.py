import argparse
from pathlib import Path

import anndata
import scanpy as sc
import wandb

import novae
from novae.monitor.log import log_plt_figure

from .utils import init_wandb_logger, read_config


def main(args: argparse.Namespace) -> None:
    config = read_config(args)

    path = Path("/gpfs/workdir/blampeyq/novae/data/_lung_robustness")
    adata1_split = sc.read_h5ad(path / "v1_split.h5ad")
    adata2_full = sc.read_h5ad(path / "v2_full.h5ad")

    adata_join = anndata.concat([adata1_split, adata2_full], join="inner")

    novae.spatial_neighbors(adata_join, slide_key="slide_id", radius=80)

    logger = init_wandb_logger(config)

    model = novae.Novae(adata_join, **config.model_kwargs)
    model.fit(logger=logger, **config.fit_kwargs)

    model.compute_representations(adata_join)
    obs_key = model.assign_domains(adata_join, level=7)

    novae.plot.domains(adata_join, obs_key=obs_key, show=False)
    log_plt_figure(f"domains_{obs_key}")

    adata1_split, adata2_split = [
        adata_join[adata_join.obs["novae_sid"] == ID] for ID in adata_join.obs["novae_sid"].unique()
    ]

    adata2_split = adata2_full[adata2_full.obsm["spatial"][:, 0] < 5000].copy()
    jsd = novae.monitor.jensen_shannon_divergence([adata1_split, adata2_split], obs_key=obs_key)

    wandb.log({"metrics/cjsd": 1 - jsd})


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
