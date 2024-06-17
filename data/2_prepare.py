import argparse
from pathlib import Path

import anndata
import scanpy as sc
from anndata import AnnData

import novae


def preprocess(adata: AnnData, compute_umap: bool = False):
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    if compute_umap:
        sc.pp.neighbors(adata, n_pcs=50)
        sc.tl.umap(adata)

    novae.utils.spatial_neighbors(adata, radius=[0, 100])


def main(args):
    data_path: Path = Path(args.path).absolute()
    out_dir = data_path / args.name

    if not out_dir.exists():
        out_dir.mkdir()

    for dataset in args.datasets:
        dataset_dir: Path = data_path / dataset
        for file in dataset_dir.glob("**/adata.h5ad"):
            print("Reading file", file)

            adata = anndata.read_h5ad(file)
            adata.obs["technology"] = dataset

            if "slide_id" not in adata.obs:
                print("    (no slide_id in obs, skipping)")
                continue

            out_file = out_dir / f"{adata.obs['slide_id'].iloc[0]}.h5ad"

            if out_file.exists() and not args.overwrite:
                print("    (already exists)")
                continue

            preprocess(adata, compute_umap=args.umap)
            adata.write_h5ad(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=".",
        help="Path to spatial directory",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="all",
        help="Name of the resulting data directory",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=["xenium", "merscope"],
        help="List of dataset names to concatenate",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "-u",
        "--umap",
        action="store_true",
        help="Whether to compute the UMAP embedding",
    )

    args = parser.parse_args()
    main(args)
