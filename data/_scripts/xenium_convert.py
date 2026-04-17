import argparse
from pathlib import Path
from typing import Any

import anndata
import h5py
import numpy as np
import pandas as pd
from anndata import AnnData


def convert_to_h5ad(dataset_dir: Path):
    res_path = dataset_dir / "adata.h5ad"

    if res_path.exists():
        print(f"File {res_path} already existing.")
        return

    adata: anndata.AnnData = _get_tables_and_circles(dataset_dir)
    adata.obs["cell_id"] = adata.obs["cell_id"].apply(lambda x: x if (isinstance(x, (str, int))) else x.decode("utf-8"))

    slide_id = dataset_dir.name
    adata.obs.index = adata.obs["cell_id"].astype(str).values + f"_{slide_id}"

    adata.obs["slide_id"] = pd.Series(slide_id, index=adata.obs_names, dtype="category")

    adata.write_h5ad(res_path)

    print(f"Created file at path {res_path}")


def _get_tables_and_circles(path: Path, gex_only: bool = True) -> AnnData | tuple[AnnData, AnnData]:
    adata = _read_10x_h5(path / "cell_feature_matrix.h5", gex_only=gex_only)
    metadata = pd.read_parquet(path / "cells.parquet")

    circ = metadata[["x_centroid", "y_centroid"]].to_numpy()
    adata.obsm["spatial"] = circ
    metadata.drop(["x_centroid", "y_centroid"], axis=1, inplace=True)
    adata.obs = metadata

    return adata


def _read_10x_h5(
    filename: str | Path,
    genome: str | None = None,
    gex_only: bool = True,
) -> AnnData:
    """Read 10x-Genomics-formatted hdf5 file.

    Parameters
    ----------
    filename
        Path to a 10x hdf5 file.
    genome
        Filter expression to genes within this genome. For legacy 10x h5
        files, this must be provided if the data contains more than one genome.
    gex_only
        Only keep 'Gene Expression' data and ignore other feature types,
        e.g. 'Antibody Capture', 'CRISPR Guide Capture', or 'Custom'

    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name.
    Stores the following information:

        - `~anndata.AnnData.X`: The data matrix is stored
        - `~anndata.AnnData.obs_names`: Cell names
        - `~anndata.AnnData.var_names`: Gene names
        - `['gene_ids']`: Gene IDs
        - `['feature_types']`: Feature types
    """
    filename = Path(filename) if isinstance(filename, str) else filename
    with h5py.File(str(filename), "r") as f:
        v3 = "/matrix" in f

    if v3:
        adata = _read_v3_10x_h5(filename)
        if genome:
            if genome not in adata.var["genome"].values:
                raise ValueError(
                    f"Could not find data corresponding to genome `{genome}` in `{filename}`. "
                    f"Available genomes are: {list(adata.var['genome'].unique())}."
                )
            adata = adata[:, adata.var["genome"] == genome]
        if gex_only:
            adata = adata[:, adata.var["feature_types"] == "Gene Expression"]
        if adata.is_view:
            adata = adata.copy()
    else:
        raise ValueError("Versions older than V3 are not supported.")
    return adata


def _read_v3_10x_h5(filename: str | Path) -> AnnData:
    """Read hdf5 file from Cell Ranger v3 or later versions."""
    with h5py.File(str(filename), "r") as f:
        try:
            dsets: dict[str, Any] = {}
            _collect_datasets(dsets, f["matrix"])

            from scipy.sparse import csr_matrix

            M, N = dsets["shape"]
            data = dsets["data"]
            if dsets["data"].dtype == np.dtype("int32"):
                data = dsets["data"].view("float32")
                data[:] = dsets["data"]
            matrix = csr_matrix(
                (data, dsets["indices"], dsets["indptr"]),
                shape=(N, M),
            )

            # Undo fixed-point scaling factor applied to Xenium Protein data
            # that is stored in HDF5.
            feature_types = dsets["feature_type"].astype(str)
            if "protein_scaling_factor" in f.attrs:
                protein_feats = np.flatnonzero(feature_types == "Protein Expression")
                if len(protein_feats) > 0:
                    matrix[:, protein_feats] /= f.attrs["protein_scaling_factor"]

            return AnnData(
                matrix,
                obs={"obs_names": dsets["barcodes"].astype(str)},
                var={
                    "var_names": dsets["name"].astype(str),
                    "gene_ids": dsets["id"].astype(str),
                    "feature_types": dsets["feature_type"].astype(str),
                    "genome": dsets["genome"].astype(str),
                },
            )
        except KeyError:
            raise Exception("File is missing one or more required datasets.") from None  # noqa: TRY002


def _collect_datasets(dsets: dict[str, Any], group: h5py.Group) -> None:
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            dsets[k] = v[:]
        else:
            _collect_datasets(dsets, v)


def main(args):
    path = Path(args.path).absolute() / "xenium"

    print(f"Reading all datasets inside {path}")

    for dataset_dir in path.iterdir():
        if dataset_dir.is_dir():
            print(f"In {dataset_dir}")
            try:
                convert_to_h5ad(dataset_dir)
            except Exception as e:
                print(f"Failed to convert {dataset_dir}. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=".",
        help="Path to spatial directory (containing the 'xenium' directory)",
    )

    main(parser.parse_args())
