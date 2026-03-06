import logging
from pathlib import Path

import anndata
import pandas as pd
from anndata import AnnData

from ..._constants import Keys

log = logging.getLogger(__name__)


def load_local_dataset(path: str, files_black_list: list[str] | None = None) -> list[AnnData]:
    """Load one or multiple AnnData objects based on a relative path from the data directory.

    Args:
        path: Path to the data directory. If a directory, loads every .h5ad files inside it. Can also be a single file. If a directory, you can add a `white_list.csv` file to mention slides to keep (have a novae_sid and novae_tissue column).

    Returns:
        A list of AnnData objects
    """
    full_path = Path(path)

    files_black_list = files_black_list or []

    if full_path.is_file():
        assert full_path.name not in files_black_list, f"File {full_path} is in the black list"
        log.info(f"Loading one adata: {full_path}")
        return [anndata.read_h5ad(full_path)]

    all_paths = [path for path in full_path.glob("*.h5ad") if path.name not in files_black_list]

    selection = None
    white_list_path: Path = path / "white_list.csv"
    if white_list_path.exists():
        selection = pd.read_csv(white_list_path, index_col=Keys.SLIDE_ID)
        all_paths = [path for path in all_paths if path.stem in selection.index]

    log.info(f"Loading {len(all_paths)} adata(s): {', '.join([str(path) for path in all_paths])}")
    adatas = []

    for path in all_paths:
        adata = anndata.read_h5ad(path)
        if selection is not None:
            adata.uns[Keys.UNS_TISSUE] = selection.loc[path.stem, Keys.UNS_TISSUE]
        adatas.append(adata)

    return adatas
