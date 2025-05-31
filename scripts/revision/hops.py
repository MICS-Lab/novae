import argparse
from pathlib import Path

import lightning.pytorch as L
import pandas as pd
import scanpy as sc

import novae
from novae.monitor import heuristic

seed_kwargs = [
    {"heads": 4, "hidden_size": 64, "temperature": 0.1, "num_layers": 12},
    {"heads": 8, "hidden_size": 128, "temperature": 0.125, "num_layers": 8},
    {"heads": 4, "hidden_size": 64, "temperature": 0.125, "num_layers": 10},
    {"heads": 8, "hidden_size": 64, "temperature": 0.1, "num_layers": 8},
    {"heads": 8, "hidden_size": 32, "temperature": 0.08, "num_layers": 10},
]


def main(args):
    path = Path(args.path)
    n_classes = args.n_classes

    adata = sc.read_h5ad(path)

    novae.spatial_neighbors(adata, radius=80)

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

                model = novae.Novae(
                    adata,
                    n_hops_local=n_hops_local,
                    n_hops_view=n_hops_view,
                    **seed_kwargs[seed],
                )

                model.fit(adata, accelerator="cuda", num_workers=4)
                model.compute_representations(adata, accelerator="cuda", num_workers=8)

                try:
                    obs_key = model.assign_domains(adata, n_domains=n_classes)
                except:
                    obs_key = model.assign_domains(adata, level=n_classes)

                _heuristic = heuristic(adata, obs_key, n_classes=n_classes)

                data["heuristic"].append(_heuristic)
                data["n_classes"].append(n_classes)
                data["n_hops_local"].append(n_hops_local)
                data["n_hops_view"].append(n_hops_view)

    data = pd.DataFrame(data)

    out_file = f"/gpfs/workdir/blampeyq/res_novae/heuristic_hops_{path.stem}_{n_classes}.csv"
    print(f"Saving to {out_file}")

    data.to_csv(out_file)


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
