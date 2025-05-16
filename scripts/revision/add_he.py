from pathlib import Path

import spatialdata
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
    path = Path("/gpfs/workdir/shared/prime/spatial/sdata_lung_s3.zarr")

    sdata = spatialdata.read_zarr(path)

    novae.data.compute_histo_embeddings(sdata, get_conch(), patch_overlap_ratio=0.6)


if __name__ == "__main__":
    main()
