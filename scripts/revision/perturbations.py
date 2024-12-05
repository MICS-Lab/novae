import anndata
import torch

import novae

adatas = [
    anndata.read_h5ad("/gpfs/workdir/blampeyq/novae/data/_perturbation/HumanBreastCancerPatient1_region_0.h5ad"),
    anndata.read_h5ad(
        "/gpfs/workdir/blampeyq/novae/data/_perturbation/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_outs.h5ad"
    ),
]

model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0", force_download=True)

# Zero-shot
print("Zero-shot")
model.compute_representations(adatas, zero_shot=True)

for level in range(7, 15):
    model.assign_domains(adatas, level=level)

for adata in adatas:
    adata.obsm["novae_latent_normal"] = adata.obsm["novae_latent"]

# Attention heterogeneity
print("Attention heterogeneity")
novae.settings.store_attention_entropy = True
for adata in adatas:
    model.compute_representations(adatas[0])
    attention_entropies = model.encoder.node_aggregation._entropies.numpy()
    adata.obs["attention_entropies"] = 0.0
    adata.obs.loc[adata.obs["neighborhood_valid"], "attention_entropies"] = attention_entropies
    model.encoder.node_aggregation._entropies = torch.tensor([], dtype=torch.float32)
novae.settings.store_attention_entropy = False

# Shuffle nodes
print("Shuffle nodes")
novae.settings.shuffle_nodes = True
model.compute_representations(adatas)

for adata in adatas:
    adata.obs["rs_shuffle"] = novae.utils.get_relative_sensitivity(adata, "novae_latent_normal", "novae_latent")
novae.settings.shuffle_nodes = False

# Edge length drop
print("Edge length drop")
for adata in adatas:
    adata.obsp["spatial_distances"].data[:] = 0.01

model.compute_representations(adatas)

for adata in adatas:
    adata.obs["rs_edge_length"] = novae.utils.get_relative_sensitivity(adata, "novae_latent_normal", "novae_latent")

# Saving results
adatas[0].write_h5ad(
    "/gpfs/workdir/blampeyq/novae/data/_perturbation/HumanBreastCancerPatient1_region_0_perturbed.h5ad"
)
adatas[1].write_h5ad(
    "/gpfs/workdir/blampeyq/novae/data/_perturbation/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_outs_perturbed.h5ad"
)
