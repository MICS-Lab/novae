from pathlib import Path

import pandas as pd
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score

import novae

DIR = Path("/gpfs/workdir/shared/prime/spatial/inter_batch_effect_simulation")

names = ["lung_st", "brain_st"]
domains = [7, 15]

novae.settings.auto_preprocessing = False


def _process_one(domain: str, name: str):
    data = {
        "domain": [],
        "name": [],
        "level": [],
        "accuracy": [],
        "ari": [],
    }

    adatas = [sc.read_h5ad(DIR / name / f"{name}.h5ad")] + [
        sc.read_h5ad(DIR / name / f"{name}_level_{i}.h5ad") for i in range(1, 6)
    ]

    for adata in adatas[1:]:
        adata.X = adata.layers["raw_counts"]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    novae.utils.spatial_neighbors(adatas, radius=80)

    for min_prototypes_ratio in [0, 0.25, 0.5, 0.75, 1]:
        model = novae.Novae(adatas, min_prototypes_ratio=min_prototypes_ratio)

        model.fit(accelerator="cuda", num_workers=8, lr=1e-4)
        model.compute_representations(accelerator="cuda", num_workers=8)

        obs_key = model.assign_domains(adatas, n_domains=domain)

        adata_reference = adatas[0]

        for i, adata in enumerate(adatas):
            if i > 0:
                del adata.X
            adata.write_h5ad(f"/gpfs/workdir/shared/prime/spatial/temp/{name}_level_{i}_domains.h5ad")

            y_ref = adata_reference.obs[obs_key]
            y_other = adata.obs[obs_key]

            keep = ~y_ref.isna()
            accuracy = (y_ref[keep] == y_other[keep]).mean()
            ari = adjusted_rand_score(y_ref[keep], y_other[keep])

            data["domain"].append(domain)
            data["name"].append(name)
            data["level"].append(i)
            data["accuracy"].append(accuracy)
            data["ari"].append(ari)

    df = pd.DataFrame(data)
    out_file = f"/gpfs/workdir/blampeyq/res_novae/batch_effect_{name}.csv"

    print(f"Saving to {out_file}")
    df.to_csv(out_file)


def main():
    for domain, name in zip(domains, names):
        _process_one(domain, name)


if __name__ == "__main__":
    main()
