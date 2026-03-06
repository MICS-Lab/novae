import logging
from pathlib import Path

import anndata
import pandas as pd
from anndata import AnnData

from ..._constants import Keys

log = logging.getLogger(__name__)


def load_local_dataset(path: str) -> list[AnnData]:
    """Load one or multiple AnnData objects based on a relative path from the data directory.

    Args:
        path: Path to the data directory. If a directory, loads every .h5ad files inside it. Can also be a single file. If a directory, you can add a `white_list.csv` file to mention slides to keep (have a novae_sid and novae_tissue column).

    Returns:
        A list of AnnData objects
    """
    path: Path = Path(path)

    if path.is_file():
        log.info(f"Loading one adata: {path}")
        return [anndata.read_h5ad(path)]

    all_paths = list(path.glob("*.h5ad"))

    selection = None
    white_list_path: Path = path / "white_list.csv"
    if white_list_path.exists():
        selection = pd.read_csv(white_list_path, index_col=Keys.SLIDE_ID)
        all_paths = [p for p in all_paths if p.stem in selection.index]

    log.info(f"Loading {len(all_paths)} adata(s): {', '.join([str(p) for p in all_paths])}")
    adatas = []

    for p in all_paths:
        adata = anndata.read_h5ad(p)
        if selection is not None:
            adata.uns[Keys.UNS_TISSUE] = selection.loc[p.stem, Keys.UNS_TISSUE]
        adatas.append(adata)

    return adatas
