# isort: skip_file
# else segmentation fault when importing concept
import pyarrow  # noqa: F401

from pathlib import Path

import pandas as pd
import scanpy as sc

import novae

RES_PATH = Path("/gpfs/workdir/blampeyq/res_novae/X_scConcept")


def main() -> None:
    paths = list(RES_PATH.glob("*.h5ad"))

    data = {
        "slide_id": [],
        "mean_distances": [],
    }

    for path in paths:
        name = path.stem
        adata = sc.read_h5ad(path)

        if "spatial_distances" not in adata.obsp:
            adata.obs["slide_id"] = name
            novae.spatial_neighbors(adata, slide_key="slide_id", radius=100)

            adata.write_h5ad(path)

        data["slide_id"].append(name)
        data["mean_distances"].append(adata.obsp["spatial_distances"].data.mean())

    df = pd.DataFrame(data)
    df.to_csv(RES_PATH / "mean_distances.csv", index=False)


if __name__ == "__main__":
    main()
