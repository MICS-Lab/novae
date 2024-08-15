import argparse
from pathlib import Path

import anndata
import pandas as pd
from scipy.sparse import csr_matrix


def convert_to_h5ad(dataset_dir: Path):
    res_path = dataset_dir / "adata.h5ad"

    if res_path.exists():
        print(f"File {res_path} already existing.")
        return

    slide_id = f"stereoseq_{dataset_dir.name}"

    h5ad_files = dataset_dir.glob(".h5ad")

    if len(h5ad_files) != 1:
        print(f"Found {len(h5ad_files)} h5ad file inside {dataset_dir}. Skipping this directory.")
        return

    adata = anndata.read_h5ad(h5ad_files[0])
    adata.X = adata.layers["raw_counts"]
    del adata.layers["raw_counts"]

    adata.obsm["spatial"] = adata.obsm["spatial"].astype(float).values
    adata.obs["slide_id"] = pd.Series(slide_id, index=adata.obs_names, dtype="category")

    adata.X = csr_matrix(adata.X)
    adata.write_h5ad(res_path)

    print(f"Created file at path {res_path}")


def main(args):
    path = Path(args.path).absolute() / "stereoseq"

    print(f"Reading all datasets inside {path}")

    for dataset_dir in path.iterdir():
        if dataset_dir.is_dir():
            convert_to_h5ad(dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=".",
        help="Path to spatial directory (containing the 'stereoseq' directory)",
    )

    main(parser.parse_args())
