import anndata
import matplotlib.pyplot as plt

import novae

adata = anndata.read_h5ad("/gpfs/workdir/blampeyq/novae/data/_mgc/MGC_merged_adata_clean_graph.h5ad")

novae.data.quantile_scaling(adata)

model = novae.Novae(adata, embedding_size=62)  # 63 proteins

model.fit()

model.compute_representations()

model.plot_domains_hierarchy(max_level=16)
plt.savefig("/gpfs/workdir/blampeyq/novae/data/_mgc/domains_hierarchy.pdf", bbox_inches="tight")

for level in range(7, 15):
    model.assign_domains(level=level)

adata.write_h5ad("/gpfs/workdir/blampeyq/novae/data/_mgc/MGC_merged_adata_clean_graph_domains.h5ad")
