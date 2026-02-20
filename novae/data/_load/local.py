import logging

import anndata
from anndata import AnnData

from ...utils import repository_root

log = logging.getLogger(__name__)


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

    all_paths = list(data_dir.glob(relative_path) if ".h5ad" in relative_path else full_path.glob("*.h5ad"))

    all_paths = [path for path in all_paths if path.name not in files_black_list]

    log.info(f"Loading {len(all_paths)} adata(s): {', '.join([str(path) for path in all_paths])}")
    return [anndata.read_h5ad(path) for path in all_paths]
