import logging
from typing import Callable

import anndata
import pandas as pd
import scanpy as sc
from anndata import AnnData

from ...utils import repository_root

log = logging.getLogger(__name__)


def _read_h5ad_from_hub(name: str, row: pd.Series):
    from huggingface_hub import hf_hub_download

    file_path = f"{row['species']}/{row['tissue']}/{name}.h5ad"
    local_file = hf_hub_download(repo_id="MICS-Lab/novae", filename=file_path, repo_type="dataset")

    return sc.read_h5ad(local_file)


def load_dataset(
    pattern: str | None = None,
    tissue: list[str] | str | None = None,
    species: list[str] | str | None = None,
    technology: list[str] | str | None = None,
    custom_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    top_k: int | None = None,
    dry_run: bool = False,
) -> list[AnnData]:
    """Automatically load slides from the Novae dataset repository.

    !!! info "Selecting slides"
        The function arguments allow to filter the slides based on the tissue, species, and name pattern.
        Internally, the function reads [this dataset metadata file](https://huggingface.co/datasets/MICS-Lab/novae/blob/main/metadata.csv) to select the slides that match the provided filters.

    Args:
        pattern: Optional pattern to match the slides names.
        tissue: Optional tissue (or tissue list) to filter the slides. E.g., `"brain", "colon"`.
        species: Optional species (or species list) to filter the slides. E.g., `"human", "mouse"`.
        technology: Optional technology (or technology list) to filter the slides. E.g., `"xenium", or "visium_hd"`.
        custom_filter: Custom filter function that takes the metadata DataFrame (see above link) and returns a boolean Series to decide which rows should be kept.
        top_k: Optional number of slides to keep. If `None`, keeps all slides.
        dry_run: If `True`, the function will only return the metadata of slides that match the filters.

    Returns:
        A list of `AnnData` objects, each object corresponds to one slide.
    """
    metadata = pd.read_csv("hf://datasets/MICS-Lab/novae/metadata.csv", index_col=0)

    FILTER_COLUMN = [("species", species), ("tissue", tissue), ("technology", technology)]
    VALID_VALUES = {column: metadata[column].unique() for column, _ in FILTER_COLUMN}

    for column, value in FILTER_COLUMN:
        if value is not None:
            values = [value] if isinstance(value, str) else value
            valid_values = VALID_VALUES[column]

            assert all(value in valid_values for value in values), (
                f"Found invalid {column} value in {values}. Valid values are {valid_values}."
            )

            metadata = metadata[metadata[column].isin(values)]

    if custom_filter is not None:
        metadata = metadata[custom_filter(metadata)]

    assert not metadata.empty, "No dataset found for the provided filters."

    if pattern is not None:
        where = metadata.index.str.match(pattern)
        assert len(where), f"No dataset found for the provided pattern ({', '.join(list(metadata.index))})."
        metadata = metadata[where]

    assert not metadata.empty, "No dataset found for the provided filters."

    if top_k is not None:
        metadata = metadata.head(top_k)

    if dry_run:
        return metadata

    log.info(f"Found {len(metadata)} h5ad file(s) matching the filters.")
    return [_read_h5ad_from_hub(name, row) for name, row in metadata.iterrows()]


def load_local_dataset(relative_path: str, files_black_list: list[str] | None = None) -> list[AnnData]:
    """Load one or multiple AnnData objects based on a relative path from the data directory

    Args:
        relative_path: Relative from from the data directory. If a directory, loads every .h5ad files inside it. Can also be a file, or a file pattern.

    Returns:
        A list of AnnData objects
    """
    data_dir = repository_root() / "data"
    full_path = data_dir / relative_path

    files_black_list = files_black_list or []

    if full_path.is_file():
        assert full_path.name not in files_black_list, f"File {full_path} is in the black list"
        log.info(f"Loading one adata: {full_path}")
        return [anndata.read_h5ad(full_path)]

    all_paths = list(data_dir.rglob(relative_path) if ".h5ad" in relative_path else full_path.rglob("*.h5ad"))

    all_paths = [path for path in all_paths if path.name not in files_black_list]

    log.info(f"Loading {len(all_paths)} adata(s): {', '.join([str(path) for path in all_paths])}")
    return [anndata.read_h5ad(path) for path in all_paths]
