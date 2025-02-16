from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc

import novae
from novae._constants import Nums

Nums.WARMUP_EPOCHS = 1
Nums.SWAV_EPSILON = 0.01

suffix = "_numproto_2"

path = Path("/gpfs/workdir/blampeyq/novae/data/_lung_robustness")

adata1_split = sc.read_h5ad(path / "v1_split.h5ad")
adata2_full = sc.read_h5ad(path / "v2_full.h5ad")
# adata2_split = sc.read_h5ad(path / "v2_split.h5ad")


adatas = [adata1_split, adata2_full]

# shared_genes = adata1_split.var_names.intersection(adata2_full.var_names)
# adata1_split = adata1_split[:, shared_genes].copy()
# adata2_full = adata2_full[:, shared_genes].copy()
# adatas = [adata1_split, adata2_full]

model = novae.Novae(
    adatas,
    num_prototypes=2000,
    heads=8,
    hidden_size=128,
    min_prototypes_ratio=0.8,
)
model.fit(lr=1e-4)
model.compute_representations()

# model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")
# model.fine_tune(adatas, min_prototypes_ratio=0.5, reference="largest")
# model.compute_representations(adatas)

obs_key = model.assign_domains(adatas, resolution=1)
obs_key = model.assign_domains(adatas, resolution=0.5)
obs_key = model.assign_domains(adatas, level=7)

model.plot_prototype_weights()
plt.savefig(path / f"prototype_weights{suffix}.pdf", bbox_inches="tight")

# model.umap_prototypes()
# plt.savefig(path / f"umap_prototypes{suffix}.png", bbox_inches="tight")

for adata in adatas:
    adata.obs[f"{obs_key}_split_ft"] = adata.obs[obs_key]

### Save

for adata, name in [(adata1_split, "v1_split"), (adata2_full, "v2_full")]:
    del adata.X
    for key in list(adata.layers.keys()):
        del adata.layers[key]
    adata.write_h5ad(path / f"{name}_res2{suffix}.h5ad")
