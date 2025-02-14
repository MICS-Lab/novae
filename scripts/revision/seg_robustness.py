import anndata

import novae

adatas = [
    anndata.read_h5ad("/gpfs/workdir/blampeyq/novae/data/_colon_seg/adata_graph.h5ad"),
    anndata.read_h5ad("/gpfs/workdir/blampeyq/novae/data/_colon_seg/adata_default_graph.h5ad"),
]

suffix = "_2"

model = novae.Novae(
    adatas,
    num_prototypes=3000,
    heads=8,
    hidden_size=128,
    min_prototypes_ratio=1,
)
model.fit()
model.compute_representations()

# model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")
# model.fine_tune(adatas)
# model.compute_representations(adatas)

for level in range(7, 15):
    model.assign_domains(adatas, level=level)

adatas[0].write_h5ad(f"/gpfs/workdir/blampeyq/novae/data/_colon_seg/adata_graph_domains{suffix}.h5ad")
adatas[1].write_h5ad(f"/gpfs/workdir/blampeyq/novae/data/_colon_seg/adata_default_graph_domains{suffix}.h5ad")
