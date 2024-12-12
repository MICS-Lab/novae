import anndata
import torch
from torch_geometric.utils import scatter, softmax

import novae
from novae._constants import Nums

adatas = [
    anndata.read_h5ad("/gpfs/workdir/blampeyq/novae/data/_perturbation/HumanBreastCancerPatient1_region_0.h5ad"),
    anndata.read_h5ad(
        "/gpfs/workdir/blampeyq/novae/data/_perturbation/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_outs.h5ad"
    ),
]

model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")

novae.utils.spatial_neighbors(adatas, radius=80)

# Zero-shot
print("Zero-shot")
model.compute_representations(adatas, zero_shot=True)

for level in range(7, 15):
    model.assign_domains(adatas, level=level)

for adata in adatas:
    adata.obsm["novae_latent_normal"] = adata.obsm["novae_latent"]

# Attention heterogeneity
print("Attention heterogeneity")
for adata in adatas:
    model._datamodule = model._init_datamodule(adata)

    with torch.no_grad():
        _entropies = torch.tensor([], dtype=torch.float32)
        gat = model.encoder.gnn

        for data_batch in model.datamodule.predict_dataloader():
            averaged_attentions_list = []

            data = data_batch["main"]
            data = model._embed_pyg_data(data)

            x = data.x

            for i, (conv, norm) in enumerate(zip(gat.convs, gat.norms)):
                x, (index, attentions) = conv(
                    x, data.edge_index, edge_attr=data.edge_attr, return_attention_weights=True
                )
                averaged_attentions = scatter(attentions.mean(1), index[0], dim_size=len(data.x), reduce="mean")
                averaged_attentions_list.append(averaged_attentions)
                if i < gat.num_layers - 1:
                    x = gat.act(x)

            attention_scores = torch.stack(averaged_attentions_list).mean(0)
            attention_scores = softmax(attention_scores / 0.01, data.batch, dim=0)
            attention_entropy = scatter(-attention_scores * (attention_scores + Nums.EPS).log2(), index=data.batch)
            _entropies = torch.cat([_entropies, attention_entropy])

    adata.obs["attention_entropies"] = 0.0
    adata.obs.loc[adata.obs["neighborhood_valid"], "attention_entropies"] = _entropies.numpy(force=True)

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

for adata in adatas:
    del adata.X
    for key in list(adata.layers.keys()):
        del adata.layers[key]

adatas[0].write_h5ad(
    "/gpfs/workdir/blampeyq/novae/data/_perturbation/HumanBreastCancerPatient1_region_0_perturbed.h5ad"
)
adatas[1].write_h5ad(
    "/gpfs/workdir/blampeyq/novae/data/_perturbation/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_outs_perturbed.h5ad"
)
