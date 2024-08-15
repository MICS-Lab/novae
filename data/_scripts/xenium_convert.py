import argparse
from pathlib import Path

import anndata
import pandas as pd
from spatialdata_io.readers.xenium import _get_tables_and_circles


def convert_to_h5ad(dataset_dir: Path):
    res_path = dataset_dir / "adata.h5ad"

    if res_path.exists():
        print(f"File {res_path} already existing.")
        return

    adata: anndata.AnnData = _get_tables_and_circles(dataset_dir, False, {"region": "region_0"})
    adata.obs["cell_id"] = adata.obs["cell_id"].apply(
        lambda x: x if (isinstance(x, str) or isinstance(x, int)) else x.decode("utf-8")
    )

    slide_id = dataset_dir.name
    adata.obs.index = adata.obs["cell_id"].astype(str).values + f"_{slide_id}"

    adata.obs["slide_id"] = pd.Series(slide_id, index=adata.obs_names, dtype="category")

    adata.write_h5ad(res_path)

    print(f"Created file at path {res_path}")


def main(args):
    path = Path(args.path).absolute() / "xenium"

    print(f"Reading all datasets inside {path}")

    for dataset_dir in path.iterdir():
        if dataset_dir.is_dir():
            print(f"In {dataset_dir}")
            try:
                convert_to_h5ad(dataset_dir)
            except:
                print(f"Failed to convert {dataset_dir}")


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
