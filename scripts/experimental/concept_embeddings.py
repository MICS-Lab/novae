# isort: skip_file
# else segmentation fault when importing concept
import pyarrow  # noqa: F401

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from anndata import AnnData
from concept import scConcept

validation = False

BLAMPEYQ = Path("/gpfs/workdir/blampeyq")
PRIME = Path("/gpfs/workdir/shared/prime")

UMAP_PATH = BLAMPEYQ / "res_novae" / "umap"

GENE_INFO = BLAMPEYQ / "gene_info.csv"

PRIME_DATASET_PATH = PRIME / "data" / "spatial" / "spatial_transcriptomics"

TRAINING_FILES = (
    list(Path("/gpfs/workdir/blampeyq/novae/data").rglob("*.h5ad")) + list((PRIME_DATASET_PATH).glob("*.h5ad")),
)
VALIDATION_FILES = list((PRIME_DATASET_PATH / "novae_validation").glob("*.h5ad"))

concept = scConcept(cache_dir=BLAMPEYQ / ".cache")
concept.load_config_and_model(model_name="corpus40M-model30M")


def run_adata(adata: AnnData, name: str, res_path: Path) -> None:
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs[["center_x", "center_y"]].values

    adata = add_gene_id(adata)
    adata.obsm["X_scConcept"] = concept.extract_embeddings(adata=adata, gene_id_column="gene_id")["cls_cell_emb"]

    save_concept_embeddings(adata, name, res_path)
    save_umap(adata, name, "X_scConcept")


def save_umap(adata: AnnData, name: str, key: str) -> None:
    sc.pp.neighbors(adata, use_rep=key)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=adata.var_names[0], vmax="p95", show=False)
    plt.savefig(UMAP_PATH / f"{name}_X_scConcept2.png", bbox_inches="tight")


def add_gene_id(adata: AnnData) -> AnnData:
    df = pd.read_csv(GENE_INFO)
    df = df[df["biotype"] == "protein_coding"].groupby("approvedSymbol").first()
    adata = adata[:, adata.var_names.intersection(df.index)].copy()
    adata.var["gene_id"] = df.loc[adata.var_names, "id"]

    return adata


def save_concept_embeddings(adata: AnnData, name: str, res_path: Path) -> None:
    adata_ = AnnData(obs=adata.obs)

    for key in ["X_scConcept", "spatial"]:
        adata_.obsm[key] = adata.obsm[key]

    for key in adata.obsp:
        if key.startswith("spatial"):
            adata_.obsp[key] = adata.obsp[key]

    adata_.write_h5ad(res_path / f"{name}.h5ad")


def main(paths: list[Path], validation: bool) -> None:
    suffix = "_validation" if validation else ""
    res_path = BLAMPEYQ / "res_novae" / f"X_scConcept2{suffix}"

    res_path.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(paths):
        name = path.stem if path.stem != "adata" else path.parent.stem

        if (res_path / f"{name}.h5ad").exists():
            print(f"Skipping {path} ({i + 1}/{len(paths)}) - already processed")
            continue

        adata = sc.read_h5ad(path)

        print(f"Processing {path} ({i + 1}/{len(paths)})")
        try:
            run_adata(adata, name, res_path)
        except Exception as e:
            print(f"Error processing {path}: {e}")


if __name__ == "__main__":
    main(TRAINING_FILES, validation=False)
    main(VALIDATION_FILES, validation=True)
