import argparse
from pathlib import Path

import anndata
import pandas as pd
from scipy.sparse import csr_matrix


def convert_to_h5ad(dataset_dir: Path):
    print(f"Reading {dataset_dir}")
    res_path = dataset_dir / "adata.h5ad"

    if res_path.exists():
        print(f"File {res_path} already existing.")
        return

    slide_id = f"cosmx_{dataset_dir.name}"

    counts_files = list(dataset_dir.glob("*exprMat_file.csv"))
    metadata_files = list(dataset_dir.glob("*metadata_file.csv"))

    if len(counts_files) != 1 or len(metadata_files) != 1:
        print(f"Did not found both exprMat and metadata csv inside {dataset_dir}. Skipping this directory.")
        return

    data = pd.read_csv(counts_files[0], index_col=[0, 1])
    obs = pd.read_csv(metadata_files[0], index_col=[0, 1])

    data.index = data.index.map(lambda x: f"{x[0]}-{x[1]}")
    obs.index = obs.index.map(lambda x: f"{x[0]}-{x[1]}")

    if len(data) != len(obs):
        cell_ids = list(set(data.index) & set(obs.index))
        data = data.loc[cell_ids]
        obs = obs.loc[cell_ids]

    obs.index = obs.index.astype(str) + f"_{slide_id}"
    data.index = obs.index

    is_gene = ~data.columns.str.lower().str.contains("SystemControl")

    adata = anndata.AnnData(data.loc[:, is_gene], obs=obs)

    adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values * 0.120280945
    adata.obs["slide_id"] = pd.Series(slide_id, index=adata.obs_names, dtype="category")

    adata.X = csr_matrix(adata.X)
    adata.write_h5ad(res_path)

    print(f"Created file at path {res_path}")


def main(args):
    path = Path(args.path).absolute() / "cosmx"

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
        help="Path to spatial directory (containing the 'cosmx' directory)",
    )

    main(parser.parse_args())
