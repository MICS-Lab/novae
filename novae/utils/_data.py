import logging
from pathlib import Path

import anndata
import numpy as np
from anndata import AnnData

from . import repository_root, spatial_neighbors, wandb_log_dir

log = logging.getLogger(__name__)


def _load_dataset(relative_path: str) -> list[AnnData]:
    """Load one or multiple AnnData objects based on a relative path from the data directory

    Args:
        relative_path: Relative from from the data directory. If a directory, loads every .h5ad files inside it. Can also be a file, or a file pattern.

    Returns:
        A list of AnnData objects
    """
    data_dir = repository_root() / "data"
    full_path = data_dir / relative_path

    if full_path.is_file():
        log.info(f"Loading one adata: {full_path}")
        return [anndata.read_h5ad(full_path)]

    if ".h5ad" in relative_path:
        all_paths = list(map(str, data_dir.rglob(relative_path)))
    else:
        all_paths = list(map(str, full_path.rglob("*.h5ad")))

    log.info(f"Loading {len(all_paths)} adata(s): {', '.join(all_paths)}")
    return [anndata.read_h5ad(path) for path in all_paths]


def dummy_dataset(
    n_obs_per_domain: int = 1000,
    n_vars: int = 100,
    n_drop: int = 20,
    n_domains: int = 4,
    n_batches: int = 3,
    batch_shift_factor: float = 0.5,
    class_shift_factor: float = 2,
) -> list[AnnData]:

    batches_shift = [batch_shift_factor * np.random.randn(n_vars) for _ in range(n_batches)]
    domains_shift = [class_shift_factor * np.random.randn(n_vars) for _ in range(n_domains)]
    loc_shift = [np.array([0, 10 * i]) for i in range(n_domains)]

    adatas = []

    for batch_index in range(n_batches):
        X_, spatial_, domains_ = [], [], []
        var_names = np.array([f"g{i}" for i in range(n_vars)])

        for domain_index in range(n_domains):
            cell_shift = np.random.randn(n_obs_per_domain, n_vars)
            X_.append(cell_shift + domains_shift[domain_index] + batches_shift[batch_index])
            spatial_.append(np.random.randn(n_obs_per_domain, 2) + loc_shift[domain_index])
            domains_.append(np.array([f"domain_{domain_index}"] * n_obs_per_domain))

        X = np.concatenate(X_, axis=0).clip(0)

        if n_drop > 0:
            var_indices = np.random.choice(n_vars, size=n_vars - n_drop, replace=False)
            X = X[:, var_indices]
            var_names = var_names[var_indices]

        adata = AnnData(X=X)

        adata.obs_names = [f"c_{batch_index}_{i}" for i in range(adata.n_obs)]
        adata.var_names = var_names
        adata.obs["domain"] = np.concatenate(domains_)
        adata.obs["slide_key"] = f"slide_{batch_index}"
        adata.obsm["spatial"] = np.concatenate(spatial_, axis=0)
        spatial_neighbors(adata, radius=[0, 3])

        adatas.append(adata)

    return adatas


def _load_wandb_artifact(name: str) -> Path:
    import wandb

    api = wandb.Api()
    artifact = api.artifact(name)

    artifact_path = wandb_log_dir() / "artifacts" / artifact.name

    if artifact_path.exists():
        log.info(f"Artifact {artifact_path} already downloaded")
    else:
        log.info(f"Downloading artifact at {artifact_path}")
        artifact.download(root=artifact_path)

    return artifact_path
