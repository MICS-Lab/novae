import matplotlib.pyplot as plt
import scanpy as sc

import novae

suffix = "_constants_fit_all3"

dir_name = "/gpfs/workdir/blampeyq/novae/data/_heterogeneous"

adatas = [
    sc.read_h5ad(f"{dir_name}/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_outs.h5ad"),
    # sc.read_h5ad(f"{dir_name}/Xenium_V1_Human_Brain_GBM_FFPE_outs.h5ad"),
    sc.read_h5ad(f"{dir_name}/Xenium_V1_hLymphNode_nondiseased_section_outs.h5ad"),
]

adatas[0].uns["novae_tissue"] = "colon"
# adatas[1].uns["novae_tissue"] = "brain"
adatas[1].uns["novae_tissue"] = "lymph_node"

novae.utils.spatial_neighbors(adatas, radius=80)

model = novae.Novae(
    adatas, scgpt_model_dir="/gpfs/workdir/blampeyq/checkpoints/scgpt/scGPT_human", num_prototypes=512, temperature=0.5
)
model.fit(max_epochs=10)
model.compute_representations()

# model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")
# model.fine_tune(adatas, min_prototypes_ratio=0.25, reference="all")
# model.compute_representations(adatas)

for level in range(7, 15):
    model.assign_domains(level=level)

model.plot_prototype_weights()
plt.savefig(f"{dir_name}/prototype_weights{suffix}.pdf", bbox_inches="tight")

for i, adata in enumerate(adatas):
    del adata.X
    for key in list(adata.layers.keys()):
        del adata.layers[key]
    adata.write_h5ad(f"{dir_name}/{i}_res{suffix}.h5ad")
