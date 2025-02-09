from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc

import novae
from novae._constants import Nums

Nums.WARMUP_EPOCHS = 3

suffix = "_su_3"

path = Path("/gpfs/workdir/blampeyq/novae/data/_lung_robustness")

adata1_split = sc.read_h5ad(path / "v1_split.h5ad")
adata2_full = sc.read_h5ad(path / "v2_full.h5ad")

adatas = [adata1_split, adata2_full]

# model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")
model = novae.Novae(
    adatas,
    scgpt_model_dir="/gpfs/workdir/blampeyq/checkpoints/scgpt/scGPT_human",
    heads=16,
    hidden_size=128,
    temperature=0.1,
    num_prototypes=1024,
    background_noise_lambda=5,
    panel_subset_size=0.8,
    unshared_prototypes_ratio=0.3,
)

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

model.compute_representations()
obs_key = model.assign_domains(adatas, level=7)

for adata in adatas:
    adata.obs[f"{obs_key}_split_ft"] = adata.obs[obs_key]

### Save

for adata, name in [(adata1_split, "v1_split"), (adata2_full, "v2_full")]:
    del adata.X
    for key in list(adata.layers.keys()):
        del adata.layers[key]
    adata.write_h5ad(path / f"{name}_res2{suffix}.h5ad")
