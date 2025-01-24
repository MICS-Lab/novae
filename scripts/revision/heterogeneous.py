import matplotlib.pyplot as plt
import scanpy as sc

import novae

suffix = "_constants"

adatas = [
    sc.read_h5ad(
        "/gpfs/workdir/blampeyq/novae/data/_heterogeneous/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_outs.h5ad"
    ),
    sc.read_h5ad("/gpfs/workdir/blampeyq/novae/data/_heterogeneous/Xenium_V1_Human_Brain_GBM_FFPE_outs.h5ad"),
]

adatas[0].uns["novae_tissue"] = "colon"
adatas[1].uns["novae_tissue"] = "brain"

novae.utils.spatial_neighbors(adatas, radius=80)

model = novae.Novae(adatas, min_prototypes_ratio=0.25)

model.fit(max_epochs=10)
model.compute_representations()

for level in range(7, 15):
    model.assign_domains(level=level)

model.plot_prototype_weights()
plt.savefig(f"/gpfs/workdir/blampeyq/novae/data/_heterogeneous/prototype_weights_{suffix}.pdf", bbox_inches="tight")

for i, adata in enumerate(adatas):
    del adata.X
    for key in list(adata.layers.keys()):
        del adata.layers[key]
    adata.write_h5ad(f"/gpfs/workdir/blampeyq/novae/data/_heterogeneous/{i}_res_{suffix}.h5ad")
