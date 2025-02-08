from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc

import novae

suffix = "_su_0"

path = Path("/gpfs/workdir/blampeyq/novae/data/_lung_robustness")

adata1_split = sc.read_h5ad(path / "v1_split.h5ad")
adata2_full = sc.read_h5ad(path / "v2_full.h5ad")

adatas = [adata1_split, adata2_full]

# model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")
model = novae.Novae(adatas)

adata_prototypes = model.swav_head._adata_prototypes()
sc.pp.neighbors(adata_prototypes)
sc.tl.umap(adata_prototypes)
sc.pl.umap(adata_prototypes, color="name", show=False)
plt.savefig(path / f"umap_prototypes{suffix}_start.png")

model.fit(max_epochs=20)

adata_prototypes = model.swav_head._adata_prototypes()
sc.pp.neighbors(adata_prototypes)
sc.tl.umap(adata_prototypes)
sc.pl.umap(adata_prototypes, color="name", show=False)
plt.savefig(path / f"umap_prototypes{suffix}.png")

# model.fine_tune(adatas, min_prototypes_ratio=0.5)
# model.compute_representations(adatas)

# obs_key = model.assign_domains(adatas, level=7)

# for adata in adatas:
#     adata.obs[f"{obs_key}_split_ft"] = adata.obs[obs_key]

### Save

# for adata, name in [(adata1_full, "v1_full"), (adata1_split, "v1_split"), (adata2_full, "v2_full")]:
#     del adata.X
#     for key in list(adata.layers.keys()):
#         del adata.layers[key]
#     adata.write_h5ad(path / f"{name}_res2{suffix}.h5ad")
