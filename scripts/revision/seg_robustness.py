import anndata

import novae

adatas = [
    anndata.read_h5ad("/gpfs/workdir/blampeyq/novae/data/_colon_seg/adata_graph.h5ad"),
    anndata.read_h5ad("/gpfs/workdir/blampeyq/novae/data/_colon_seg/adata_default_graph.h5ad"),
]

model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")

model.fine_tune(adatas)
model.compute_representations(adatas)

for level in range(7, 15):
    model.assign_domains(adatas, level=level)

adatas[0].write_h5ad("/gpfs/workdir/blampeyq/novae/data/_colon_seg/adata_graph_domains.h5ad")
adatas[1].write_h5ad("/gpfs/workdir/blampeyq/novae/data/_colon_seg/adata_default_graph_domains.h5ad")
