import spatialdata
from sopa.segmentation.shapes import _ensure_polygon
from torchvision import transforms
from transformers import AutoModel

import novae


def get_conch():
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    conch, _ = titan.return_conch()
    transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    class Conch:
        def __call__(self, x):
            return conch(transform(x))

    return Conch()


def main():
    sdata = spatialdata.read_zarr("/gpfs/workdir/shared/prime/spatial/sdata_lung_s3.zarr")

    for geo_df in sdata.shapes.values():
        geo_df.geometry = geo_df.geometry.map(_ensure_polygon)

    novae.data.compute_histo_embeddings(
        sdata, get_conch(), patch_overlap_ratio=0.6, table_key="table_nuclei", image_key="he"
    )


if __name__ == "__main__":
    main()
