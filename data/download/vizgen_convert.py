import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def convert_to_h5ad(dataset_dir: Path):
    res_path = dataset_dir / "adata.h5ad"

    if res_path.exists():
        print(f"File {res_path} already existing.")
        return

    region = "region_0"
    slide = dataset_dir.name
    dataset_id = f"{dataset_dir.name}_{region}"

    data_dir = dataset_dir / "cell_by_gene.csv"
    obs_dir = dataset_dir / "cell_metadata.csv"

    if not data_dir.exists() or not obs_dir.exists():
        print(f"Did not found both csv inside {dataset_dir}. Skipping this directory.")
        return

    data = pd.read_csv(data_dir, index_col=0, dtype={"cell": str})
    obs = pd.read_csv(obs_dir, index_col=0, dtype={"EntityID": str})

    obs.index = obs.index.astype(str) + f"_{dataset_id}"
    data.index = obs.index

    is_gene = ~data.columns.str.lower().str.contains("blank")

    adata = anndata.AnnData(data.loc[:, is_gene], dtype=np.uint16, obs=obs)

    adata.obsm["spatial"] = adata.obs[["center_x", "center_y"]].values
    adata.obs["region"] = pd.Series(region, index=adata.obs_names, dtype="category")
    adata.obs["slide"] = pd.Series(slide, index=adata.obs_names, dtype="category")
    adata.obs["dataset_id"] = pd.Series(dataset_id, index=adata.obs_names, dtype="category")

    adata.X = csr_matrix(adata.X)
    adata.write_h5ad(res_path)

    print(f"Created file at path {res_path}")


def main(args):
    path = Path(args.path).absolute() / "vizgen"

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
        help="Path to spatial directory (containing the 'vizgen' directory)",
    )

    main(parser.parse_args())
