# isort: skip_file
# else segmentation fault when importing concept
import pyarrow  # noqa: F401

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from anndata import AnnData
from concept import scConcept

validation = True

BLAMPEYQ = Path("/gpfs/workdir/blampeyq")
PRIME = Path("/gpfs/workdir/shared/prime")

suffix = "_validation" if validation else ""
UMAP_PATH = BLAMPEYQ / "res_novae" / "umap"
RES_PATH = BLAMPEYQ / "res_novae" / f"X_scConcept{suffix}"

GENE_INFO = BLAMPEYQ / "gene_info.csv"
DATASET_PATH = PRIME / "data" / "spatial" / "spatial_transcriptomics"
if validation:
    DATASET_PATH = DATASET_PATH / "novae_validation"

concept = scConcept(cache_dir=BLAMPEYQ / ".cache")
concept.load_config_and_model(model_name="Corpus-30M")


def run_adata(adata: AnnData, name: str) -> None:
    adata = add_gene_id(adata)
    adata.obsm["X_scConcept"] = concept.extract_embeddings(adata=adata, gene_id_column="gene_id")["cls_cell_emb"]

    save_concept_embeddings(adata, name)
    save_umap(adata, name, "X_scConcept")


def save_umap(adata: AnnData, name: str, key: str) -> None:
    sc.pp.neighbors(adata, use_rep=key)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=adata.var_names[0], vmax="p95", show=False)
    plt.savefig(UMAP_PATH / f"{name}_{key}.png", bbox_inches="tight")


def add_gene_id(adata: AnnData) -> AnnData:
    df = pd.read_csv(GENE_INFO)
    df = df[df["biotype"] == "protein_coding"].groupby("approvedSymbol").first()
    adata = adata[:, adata.var_names.intersection(df.index)].copy()
    adata.var["gene_id"] = df.loc[adata.var_names, "id"]

    return adata


def save_concept_embeddings(adata: AnnData, name: str) -> None:
    adata_ = AnnData(obs=adata.obs)

    for key in ["X_scConcept", "spatial"]:
        adata_.obsm[key] = adata.obsm[key]

    for key in adata.obsp:
        if key.startswith("spatial"):
            adata_.obsp[key] = adata.obsp[key]

    adata_.write_h5ad(RES_PATH / f"{name}.h5ad")


def main() -> None:
    paths = list(DATASET_PATH.glob("*.h5ad"))

    for i, path in enumerate(paths):
        if (RES_PATH / f"{path.stem}.h5ad").exists():
            print(f"Skipping {path.name} ({i + 1}/{len(paths)}) - already processed")
            continue

        adata = sc.read_h5ad(path)

        print(f"Processing {path.name} ({i + 1}/{len(paths)})")
        try:
            run_adata(adata, path.stem)
        except Exception as e:
            print(f"Error processing {path.name}: {e}")


if __name__ == "__main__":
    main()
