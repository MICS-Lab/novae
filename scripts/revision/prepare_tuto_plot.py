import matplotlib.pyplot as plt
import scanpy as sc

import novae


def main():
    # output_file = "/gpfs/workdir/shared/prime/spatial/tuto.zarr"

    # sdata = spatialdata.read_zarr(output_file)
    # print(sdata)

    # adata_conch = sdata["conch_embeddings"]

    # sc.pl.spatial(adata_conch, color="0", spot_size=10, show=False)
    # plt.savefig("/gpfs/workdir/blampeyq/res_novae/conch_emb.png", bbox_inches="tight")

    # novae.compute_histo_pca(sdata)

    adata = sc.read_h5ad("/gpfs/workdir/blampeyq/res_novae/tuto_pp.h5ad")

    adata.obs["pca_dim0_cells"] = adata.obsm["histo_embeddings"][:, 0]
    sc.pl.spatial(adata, color="pca_dim0_cells", spot_size=10, show=False)
    plt.savefig("/gpfs/workdir/blampeyq/res_novae/conch_emb_projected.png", bbox_inches="tight")

    novae.spatial_neighbors(adata, radius=80)

    model = novae.Novae(adata)
    model.fit(adata, accelerator="cuda", num_workers=8)
    model.compute_representations(adata, accelerator="cuda", num_workers=8)

    model.assign_domains(adata, level=8)

    adata.write_h5ad("/gpfs/workdir/blampeyq/res_novae/tuto_res2.h5ad")


if __name__ == "__main__":
    main()
