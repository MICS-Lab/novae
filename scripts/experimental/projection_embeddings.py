from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from torch.nn import functional as F

import novae

BLAMPEYQ = Path("/gpfs/workdir/blampeyq")
PRIME = Path("/gpfs/workdir/shared/prime")

DATASET_PATH = PRIME / "data" / "spatial" / "spatial_transcriptomics"
RES_PATH = BLAMPEYQ / "res_novae"
GENE_INFO = BLAMPEYQ / "gene_info.csv"

novae_model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")


def run_adata(adata: AnnData, name: str) -> None:
    adata.obsm["novae_projection"] = get_novae_projection(adata)
    save_umap(adata, name, "novae_projection")


def save_umap(adata: AnnData, name: str, key: str) -> None:
    sc.pp.neighbors(adata, use_rep=key)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=adata.var_names[0], vmax="p95", show=False)
    plt.savefig(RES_PATH / "umap" / f"{name}_{key}.png", bbox_inches="tight")


@torch.no_grad()
def get_novae_projection(adata: AnnData) -> np.ndarray:
    datamodule = novae_model._init_datamodule(novae_model._prepare_adatas(adata))

    X = datamodule.dataset.torch_converter.tensors[0]
    genes_indices_list = datamodule.dataset.torch_converter.genes_indices_list[0]

    genes_embeddings = novae_model.cell_embedder.embedding(genes_indices_list)
    genes_embeddings = novae_model.cell_embedder.linear(genes_embeddings)
    genes_embeddings = F.normalize(genes_embeddings, dim=0, p=2)

    return (X @ genes_embeddings).cpu().numpy()[0]


def main() -> None:
    paths = list(DATASET_PATH.glob("*.h5ad"))

    for i, path in enumerate(paths):
        adata = sc.read_h5ad(path)

        print(f"Processing {path.name} ({i + 1}/{len(paths)})")
        run_adata(adata, path.stem)


if __name__ == "__main__":
    main()
