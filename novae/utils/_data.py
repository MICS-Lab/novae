import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
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
    n_panels: int = 3,
    n_slides_per_panel: int = 1,
    panel_shift_factor: float = 0.5,
    batch_shift_factor: float = 0.2,
    class_shift_factor: float = 2,
    slide_ids_unique: bool = True,
    compute_spatial_neighbors: bool = True,
) -> list[AnnData]:
    """Creates a dummy dataset, useful for debugging or testing.

    Args:
        n_obs_per_domain: Number of obs per domain or niche.
        n_vars: Number of genes.
        n_drop: Number of genes that are removed for each `AnnData` object. It will create non-identical panels.
        n_domains: Number of domains, or niches.
        n_panels: Number of panels. Each panel will correspond to one output `AnnData` object.
        n_slides_per_panel: Number of slides per panel.
        panel_shift_factor: Shift factor for each panel.
        batch_shift_factor: Shift factor for each batch.
        class_shift_factor: Shift factor for each niche.
        slide_ids_unique: Whether to ensure that slide ids are unique.
        compute_spatial_neighbors: Whether to compute the spatial neighbors graph.

    Returns:
        A list of `AnnData` objects representing a valid `Novae` dataset.
    """
    assert n_obs_per_domain > 10
    assert n_vars - n_drop - n_panels > 2

    panels_shift = [panel_shift_factor * np.random.randn(n_vars) for _ in range(n_panels)]
    domains_shift = [class_shift_factor * np.random.randn(n_vars) for _ in range(n_domains)]
    loc_shift = [np.array([0, 10 * i]) for i in range(n_domains)]

    adatas = []

    for panel_index in range(n_panels):
        X_, spatial_, domains_, slide_ids_ = [], [], [], []
        var_names = np.array([f"g{i}" for i in range(n_vars)])

        slide_key = f"slide_{panel_index}_" if slide_ids_unique else "slide_"
        if n_slides_per_panel > 1:
            slides_shift = np.array([batch_shift_factor * np.random.randn(n_vars) for _ in range(n_slides_per_panel)])

        for domain_index in range(n_domains):
            n_obs = n_obs_per_domain + panel_index  # ensure n_obs different
            cell_shift = np.random.randn(n_obs, n_vars)
            slide_ids_domain_ = np.random.randint(0, n_slides_per_panel, n_obs)
            X_domain_ = cell_shift + domains_shift[domain_index] + panels_shift[panel_index]

            if n_slides_per_panel > 1:
                X_domain_ += slides_shift[slide_ids_domain_]

            X_.append(X_domain_)
            spatial_.append(np.random.randn(n_obs, 2) + loc_shift[domain_index])
            domains_.append(np.array([f"domain_{domain_index}"] * n_obs))
            slide_ids_.append(slide_ids_domain_)

        X = np.concatenate(X_, axis=0).clip(0)

        if n_drop > 0:
            size = n_vars - n_drop - panel_index  # ensure n_vars different
            var_indices = np.random.choice(n_vars, size=size, replace=False)
            X = X[:, var_indices]
            var_names = var_names[var_indices]

        adata = AnnData(X=X)

        adata.obs_names = [f"c_{panel_index}_{i}" for i in range(adata.n_obs)]
        adata.var_names = var_names
        adata.obs["domain"] = np.concatenate(domains_)
        adata.obs["slide_key"] = (slide_key + pd.Series(np.concatenate(slide_ids_)).astype(str)).values
        adata.obsm["spatial"] = np.concatenate(spatial_, axis=0)

        adata.obsm["spatial"][[1, 2]] += 100  # ensure at least two indices are connected to only one node
        adata.obsm["spatial"][4] += 1000  # ensure at least one index is non-connected

        if compute_spatial_neighbors:
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
