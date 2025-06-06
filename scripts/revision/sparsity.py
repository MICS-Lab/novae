import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score

import novae


def main(args):
    path = Path(args.path)
    n_classes = args.n_classes

    adata = sc.read_h5ad(path)

    novae.spatial_neighbors(adata, radius=80)

    data = {
        "ari": [],
        "dropout": [],
        "n_cells": [],
        "accuracy": [],
    }
    model = novae.Novae(adata)

    model.fit(adata, accelerator="cuda", num_workers=4)
    model.compute_representations(adata, accelerator="cuda", num_workers=8)
    obs_key = model.assign_domains(adata, n_domains=n_classes)

    y_ref = adata.obs[obs_key].astype(str)

    for dropout in [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]:
        n_cells = int(adata.n_obs * dropout)
        adata_ = adata[np.random.choice(adata.n_obs, n_cells, replace=False)].copy()

        model.fit(adata_, accelerator="cuda", num_workers=4)
        model.compute_representations(adata_, accelerator="cuda", num_workers=8)

        try:
            obs_key = model.assign_domains(adata_, n_domains=n_classes)
        except:
            print("Failed to compute domains")
            obs_key = model.assign_domains(adata_, level=n_classes)

        y_other = adata_.obs[obs_key].astype(str)

        keep = ~y_ref.isna()
        accuracy = (y_ref[keep] == y_other[keep]).mean()
        ari = adjusted_rand_score(y_ref[keep], y_other[keep])

        data["accuracy"].append(accuracy)
        data["ari"].append(ari)
        data["n_cells"].append(n_cells)
        data["dropout"].append(dropout)

        print(data)

    data = pd.DataFrame(data)

    out_file = f"/gpfs/workdir/blampeyq/res_novae/sparsity_{path.stem}_{n_classes}.csv"
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
