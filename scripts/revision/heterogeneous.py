import matplotlib.pyplot as plt
import scanpy as sc

import novae

suffix = "_constants_ft"

dir_name = "/gpfs/workdir/blampeyq/novae/data/_heterogeneous"

adatas = [
    sc.read_h5ad(f"{dir_name}/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_outs.h5ad"),
    # sc.read_h5ad(f"{dir_name}/Xenium_V1_Human_Brain_GBM_FFPE_outs.h5ad"),
    sc.read_h5ad(f"{dir_name}/Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs.h5ad"),
]

adatas[0].uns["novae_tissue"] = "colon"
# adatas[1].uns["novae_tissue"] = "brain"
adatas[1].uns["novae_tissue"] = "bone_marrow"

novae.utils.spatial_neighbors(adatas, radius=80)

model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")

model.fine_tune(adatas, min_prototypes_ratio=0.25)
model.compute_representations(adatas)

# model.fit(max_epochs=10)
# model.compute_representations()

for level in range(7, 15):
    model.assign_domains(level=level)

model.plot_prototype_weights()
plt.savefig(f"{dir_name}/prototype_weights{suffix}.pdf", bbox_inches="tight")

for i, adata in enumerate(adatas):
    del adata.X
    for key in list(adata.layers.keys()):
        del adata.layers[key]
    adata.write_h5ad(f"{dir_name}/{i}_res{suffix}.h5ad")
