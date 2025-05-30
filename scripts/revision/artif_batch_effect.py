from pathlib import Path

import scanpy as sc

import novae

DIR = Path("/gpfs/workdir/shared/prime/spatial/inter_batch_effect_simulation")

names = ["lung_st", "brain_st"]
domains = [7, 15]


def main():
    total = []

    for domain, name in zip(domains, names):
        adatas = [sc.read_h5ad(DIR / name / f"{name}.h5ad")] + [
            sc.read_h5ad(DIR / name / f"{name}_level_{i}.h5ad") for i in range(1, 6)
        ]

        for adata in adatas[1:]:
            adata.X = adata.layers["raw_counts"]
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)

        novae.utils.spatial_neighbors(adatas, radius=80)

        model = novae.Novae(adatas)

        model.fit(accelerator="cuda", num_workers=4)
        model.compute_representations(accelerator="cuda", num_workers=8)

        obs_key = model.assign_domains(adatas, n_domains=domain)

        adata_reference = adatas[0]

        scores = []
        for adata in adatas[1:]:
            gt = adata_reference.obs[obs_key]
            pred = adata.obs[obs_key]

            keep = ~gt.isna()
            accuracy = (gt[keep] == pred[keep]).mean()
            scores.append(accuracy)

        for i, adata in enumerate(adatas):
            if i > 0:
                del adata.X

            adata.write_h5ad(f"/gpfs/workdir/shared/prime/spatial/temp/{name}_level_{i}_domains.h5ad")

        _res = f"Domain: {domain}, Name: {name}, Scores: {scores}"

        print(_res)
        total.append(_res)

    print(total)


if __name__ == "__main__":
    main()
