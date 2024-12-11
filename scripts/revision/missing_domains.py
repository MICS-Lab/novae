from pathlib import Path

import scanpy as sc

import novae

path = Path("/gpfs/workdir/blampeyq/novae/data/_lung_robustness")

adata1_full = sc.read_h5ad(path / "v1_full.h5ad")
adata1_split = sc.read_h5ad(path / "v1_split.h5ad")
adata2_full = sc.read_h5ad(path / "v2_full.h5ad")
adata2_split = sc.read_h5ad(path / "v2_split.h5ad")

model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")

### Full

adatas = [adata1_full, adata2_full]

model.compute_representations(adatas, zero_shot=True)
obs_key = model.assign_domains(adatas, level=7)

for adata in adatas:
    adata.obs[f"{obs_key}_full"] = adata.obs[obs_key]

### Split

adatas = [adata1_split, adata2_full]

model.compute_representations(adatas, zero_shot=True)
obs_key = model.assign_domains(adatas, level=7)

for adata in adatas:
    adata.obs[f"{obs_key}_split"] = adata.obs[obs_key]

### Save

for adata, name in [(adata1_full, "v1_full"), (adata1_split, "v1_split"), (adata2_full, "v2_full")]:
    del adata.X
    for key in list(adata.layers.keys()):
        del adata.layers[key]
    adata.write_h5ad(path / f"{name}_res.h5ad")
