import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score

import novae

DIR = Path("/gpfs/workdir/shared/prime/spatial/inter_batch_effect_simulation")

names = ["lung_st", "brain_st"]


def main(args):
    name = args.name
    domain = 7 if name == "lung_st" else 15

    data = {
        "domain": [],
        "name": [],
        "level": [],
        "accuracy": [],
        "ari": [],
        "min_prototypes_ratio": [],
    }

    adatas = [sc.read_h5ad(DIR / name / f"{name}.h5ad")] + [
        sc.read_h5ad(DIR / name / f"{name}_level_{i}.h5ad") for i in range(1, 6)
    ]

    for adata in adatas[1:]:
        adata.X = adata.layers["raw_counts"]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata.X = adata.X.clip(0, 9)

    print("max values:", [adata.X.max() for adata in adatas])

    novae.utils.spatial_neighbors(adatas, radius=80)

    adata_reference = adatas[0]

    for min_prototypes_ratio in [0, 0.33, 0.67, 1]:
        for i, adata in enumerate(adatas[1:]):
            adatas_ = [adata_reference, adata]
            model = novae.Novae(adatas_, min_prototypes_ratio=min_prototypes_ratio, temperature=args.temperature)

            model.fit(accelerator="cuda", num_workers=8, lr=1e-4)
            model.compute_representations(accelerator="cuda", num_workers=8)

            obs_key = model.assign_domains(adatas_, n_domains=domain)

            adata_reference.obs[f"domains_ref_{min_prototypes_ratio}"] = adata_reference.obs[obs_key]

            adata_reference.obs[f"domains_level{i}_{min_prototypes_ratio}"] = adata.obs[obs_key]

            y_ref = adata_reference.obs[obs_key].astype(str)
            y_other = adata.obs[obs_key].astype(str)

            keep = ~y_ref.isna()
            accuracy = (y_ref[keep] == y_other[keep]).mean()
            ari = adjusted_rand_score(y_ref[keep], y_other[keep])

            data["domain"].append(domain)
            data["name"].append(name)
            data["level"].append(i + 1)
            data["accuracy"].append(accuracy)
            data["ari"].append(ari)
            data["min_prototypes_ratio"].append(min_prototypes_ratio)

        adata_reference.write_h5ad(
            f"/gpfs/workdir/shared/prime/spatial/temp/{name}_{min_prototypes_ratio}_domains2.h5ad"
        )

    df = pd.DataFrame(data)
    out_file = f"/gpfs/workdir/blampeyq/res_novae/batch_effect2_{name}.csv"

    print(f"Saving to {out_file}")
    df.to_csv(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Name of the expe",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.1,
        help="Model temperature",
    )

    main(parser.parse_args())
