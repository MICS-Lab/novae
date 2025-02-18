from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import scanpy as sc

import novae
from novae._constants import Nums

Nums.WARMUP_EPOCHS = 1
# Nums.SWAV_EPSILON = 0.01

suffix = "_numproto_5"

path = Path("/gpfs/workdir/blampeyq/novae/data/_lung_robustness")

adata1_split = sc.read_h5ad(path / "v1_split.h5ad")
adata2_full = sc.read_h5ad(path / "v2_full.h5ad")
# adata2_split = sc.read_h5ad(path / "v2_split.h5ad")

adata_join = anndata.concat([adata1_split, adata2_full], join="inner")
# adatas = [adata1_split, adata2_full]

novae.spatial_neighbors(adata_join, slide_key="slide_id", radius=80)

# shared_genes = adata1_split.var_names.intersection(adata2_full.var_names)
# adata1_split = adata1_split[:, shared_genes].copy()
# adata2_full = adata2_full[:, shared_genes].copy()
# adatas = [adata1_split, adata2_full]

model = novae.Novae(
    adata_join,
    num_prototypes=2000,
    heads=8,
    hidden_size=128,
    min_prototypes_ratio=0.5,
)
model.fit()
model.compute_representations()

# model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")
# model.fine_tune(adatas, min_prototypes_ratio=0.5, reference="largest")
# model.compute_representations(adatas)

obs_key = model.assign_domains(adata_join, resolution=1)
for res in [0.3, 0.35, 0.4, 0.45, 0.5]:
    obs_key = model.assign_domains(adata_join, resolution=res)
obs_key = model.assign_domains(adata_join, level=7)

model.plot_prototype_weights()
plt.savefig(path / f"prototype_weights{suffix}.pdf", bbox_inches="tight")

# model.umap_prototypes()
# plt.savefig(path / f"umap_prototypes{suffix}.png", bbox_inches="tight")

adatas = [adata_join[adata_join.obs["novae_sid"] == ID] for ID in adata_join.obs["novae_sid"].unique()]
names = ["v1_split", "v2_full"]

### Save

for adata, name in zip(adatas, names):
    adata.obs[f"{obs_key}_split_ft"] = adata.obs[obs_key]
    del adata.X
    for key in list(adata.layers.keys()):
        del adata.layers[key]
    adata.write_h5ad(path / f"{name}_res2{suffix}.h5ad")
