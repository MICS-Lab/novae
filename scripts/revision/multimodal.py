import scanpy as sc

import novae


def main():
    adata = sc.read_h5ad("/gpfs/workdir/blampeyq/data/adata_lung_conch_cells.h5ad")
    novae.spatial_neighbors(adata)

    # scaler = StandardScaler()
    # adata.obsm["histo_embeddings"] = scaler.fit_transform(adata.obsm["histo_embeddings"])

    # model = novae.Novae(adata)
    # model.fit(adata, accelerator="cuda", num_workers=4)
    # model.compute_representations(adata, accelerator="cuda", num_workers=4)

    # model.assign_domains(adata, level=8)

    # adata.write_h5ad("/gpfs/workdir/blampeyq/res_novae/adata_lung_conch.h5ad")

    del adata.obsm["histo_embeddings"]

    model = novae.Novae(adata)
    model.fit(adata, accelerator="cuda", num_workers=4)
    model.compute_representations(adata, accelerator="cuda", num_workers=4)

    model.assign_domains(adata, level=8)

    adata.write_h5ad("/gpfs/workdir/blampeyq/res_novae/adata_lung_no_conch.h5ad")


if __name__ == "__main__":
    main()
