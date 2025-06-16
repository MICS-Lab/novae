import spatialdata

import novae


def main():
    # input_file = "/gpfs/workdir/shared/prime/spatial/spatialdata"
    output_file = "/gpfs/workdir/shared/prime/spatial/tuto.zarr"

    # sdata.write(output_file)
    # sdata = sopa.io.xenium(input_file, cells_table=True, cells_boundaries=True)

    sdata = spatialdata.read_zarr(output_file)
    print(sdata)

    novae.compute_histo_embeddings(sdata, device="cuda")
    novae.compute_histo_pca(sdata)

    adata = sdata["table"]

    adata.write_h5ad("/gpfs/workdir/blampeyq/res_novae/tuto_pp.h5ad")

    novae.spatial_neighbors(adata, radius=80)

    model = novae.Novae(adata)
    model.fit(adata, accelerator="cuda", num_workers=4, lr=2e-4, max_epochs=40)
    model.compute_representations(adata, accelerator="cuda", num_workers=4)

    model.assign_domains(adata, level=8)

    adata.write_h5ad("/gpfs/workdir/blampeyq/res_novae/tuto_res.h5ad")


if __name__ == "__main__":
    main()
