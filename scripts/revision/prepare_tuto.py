import sopa
import spatialdata

import novae


def main():
    input_file = "/gpfs/workdir/shared/prime/spatial/spatialdata"
    output_file = "/gpfs/workdir/shared/prime/spatial/tuto.zarr"

    sdata = sopa.io.xenium(input_file, cells_table=True)
    print(sdata)

    sdata.write(output_file)
    sdata = spatialdata.read_zarr(output_file)

    novae.compute_histo_embeddings(sdata, device="cuda")
    novae.compute_histo_pca(sdata)

    sdata["table"].write_h5ad("/gpfs/workdir/blampeyq/res_novae/tuto_pp.h5ad")


if __name__ == "__main__":
    main()
