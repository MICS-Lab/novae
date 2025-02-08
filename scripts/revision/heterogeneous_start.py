import matplotlib.pyplot as plt
import scanpy as sc

import novae
from novae._constants import Nums

Nums.QUEUE_WEIGHT_THRESHOLD_RATIO = 0.99

suffix = ""

dir_name = "/gpfs/workdir/blampeyq/novae/data/_heterogeneous"

adatas = [
    sc.read_h5ad(f"{dir_name}/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_outs.h5ad"),
    # sc.read_h5ad(f"{dir_name}/Xenium_V1_Human_Brain_GBM_FFPE_outs.h5ad"),
    sc.read_h5ad(f"{dir_name}/Xenium_V1_hLymphNode_nondiseased_section_outs.h5ad"),
]

adatas[0].uns["novae_tissue"] = "colon"
# adatas[1].uns["novae_tissue"] = "brain"
adatas[1].uns["novae_tissue"] = "lymph_node"

for adata in adatas:
    adata.obs["novae_tissue"] = adata.uns["novae_tissue"]

novae.utils.spatial_neighbors(adatas, radius=80)

# model = novae.Novae(adatas)
# model.mode.trained = True
# model.compute_representations(adatas)
# model.assign_domains(adatas)

# adata = sc.concat(adatas, join="inner")
# adata = sc.pp.subsample(adata, n_obs=100_000, copy=True)
# sc.pp.neighbors(adata, use_rep="novae_latent")
# sc.tl.umap(adata)
# sc.pl.umap(adata, color=["novae_domains_7", "novae_tissue"])
# plt.savefig(f"{dir_name}/umap_start{suffix}.png", bbox_inches="tight")

model = novae.Novae(
    adatas,
    scgpt_model_dir="/gpfs/workdir/blampeyq/checkpoints/scgpt/scGPT_human",
)
model.mode.trained = True
model.compute_representations(adatas)
model.assign_domains(adatas)

adata = sc.concat(adatas, join="inner")
adata = sc.pp.subsample(adata, n_obs=100_000, copy=True)
sc.pp.neighbors(adata, use_rep="novae_latent")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["novae_domains_7", "novae_tissue"])
plt.savefig(f"{dir_name}/umap_start_scgpt{suffix}.png", bbox_inches="tight")
