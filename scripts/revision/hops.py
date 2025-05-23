import argparse
from pathlib import Path

import lightning.pytorch as L
import pandas as pd
import scanpy as sc

import novae
from novae.monitor import heuristic


def main(args):
    path = Path(args.path)
    n_classes = args.n_classes

    adata = sc.read_h5ad(path)

    data = {
        "heuristic": [],
        "n_classes": [],
        "n_hops_local": [],
        "n_hops_view": [],
    }

    for n_hops_local in [1, 2, 3]:
        for n_hops_view in [1, 2, 3]:
            for seed in range(5):
                L.seed_everything(seed)

                model = novae.Novae(adata, n_hops_local=n_hops_local, n_hops_view=n_hops_view)

                model.fit(adata, accelerator="cuda")
                model.compute_representations(adata, accelerator="cuda")

                obs_key = model.assign_domains(adata, n_domains=n_classes)
                _heuristic = heuristic(adata, obs_key, n_classes=n_classes)

                data["heuristic"].append(_heuristic)
                data["n_classes"].append(n_classes)
                data["n_hops_local"].append(n_hops_local)
                data["n_hops_view"].append(n_hops_view)

    data = pd.DataFrame(data)
    data.to_csv(path.parent / f"heuristic_{path.stem}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to the h5ad file",
    )
    parser.add_argument(
        "-n",
        "--n_classes",
        type=int,
        default=7,
        help="Number of classes to use for the heuristic",
    )

    main(parser.parse_args())
