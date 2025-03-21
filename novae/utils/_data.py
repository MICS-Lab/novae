import logging
from pathlib import Path
from typing import Callable

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from . import repository_root, spatial_neighbors, wandb_log_dir

log = logging.getLogger(__name__)


def toy_dataset(
    n_panels: int = 3,
    n_domains: int = 4,
    n_slides_per_panel: int = 1,
    xmax: int = 500,
    n_vars: int = 100,
    n_drop: int = 20,
    step: int = 20,
    panel_shift_lambda: float = 5,
    slide_shift_lambda: float = 1.5,
    domain_shift_lambda: float = 2.0,
    slide_ids_unique: bool = True,
    compute_spatial_neighbors: bool = False,
    merge_last_domain_even_slide: bool = False,
) -> list[AnnData]:
    """Creates a toy dataset, useful for debugging or testing.

    Args:
        n_panels: Number of panels. Each panel will correspond to one output `AnnData` object.
        n_domains: Number of domains.
        n_slides_per_panel: Number of slides per panel.
        xmax: Maximum value for the spatial coordinates (the larger, the more cells).
        n_vars: Maxmium number of genes per panel.
        n_drop: Number of genes that are randomly removed for each `AnnData` object. It will create non-identical panels.
        step: Step between cells in their spatial coordinates.
        panel_shift_lambda: Lambda used in the exponential law for each panel.
        slide_shift_lambda: Lambda used in the exponential law for each slide.
        domain_shift_lambda: Lambda used in the exponential law for each domain.
        slide_ids_unique: Whether to ensure that slide ids are unique.
        compute_spatial_neighbors: Whether to compute the spatial neighbors graph. We remove some the edges of one node for testing purposes.

    Returns:
        A list of `AnnData` objects representing a valid `Novae` dataset.
    """
    assert n_vars - n_drop - n_panels > 2

    spatial = np.mgrid[-xmax:xmax:step, -xmax:xmax:step].reshape(2, -1).T
    spatial = spatial[(spatial**2).sum(1) <= xmax**2]
    n_obs = len(spatial)

    int_domains = (np.sqrt((spatial**2).sum(1)) // (xmax / n_domains + 1e-8)).astype(int)
    domain = "domain_" + int_domains.astype(str).astype(object)
    merge_domain = "domain_" + int_domains.clip(0, n_domains - 2).astype(str).astype(object)

    adatas = []

    var_names = np.array(
        GENE_NAMES_SUBSET[:n_vars] if n_vars <= len(GENE_NAMES_SUBSET) else [f"g{i}" for i in range(n_vars)]
    )

    domains_shift = np.random.exponential(domain_shift_lambda, size=(n_domains, n_vars))

    for panel_index in range(n_panels):
        adatas_panel = []
        panel_shift = np.random.exponential(panel_shift_lambda, size=n_vars)

        for slide_index in range(n_slides_per_panel):
            slide_shift = np.random.exponential(slide_shift_lambda, size=n_vars)

            merge = merge_last_domain_even_slide and (slide_index % 2 == 0)

            adata = AnnData(
                np.zeros((n_obs, n_vars)),
                obsm={"spatial": spatial + panel_index + slide_index},  # ensure the locs are different
                obs=pd.DataFrame(
                    {"domain": merge_domain if merge else domain}, index=[f"cell_{i}" for i in range(spatial.shape[0])]
                ),
            )

            adata.var_names = var_names
            adata.obs_names = [f"c_{panel_index}_{slide_index}_{i}" for i in range(adata.n_obs)]

            slide_key = f"slide_{panel_index}_{slide_index}" if slide_ids_unique else f"slide_{slide_index}"
            adata.obs["slide_key"] = slide_key

            for i in range(n_domains):
                condition = adata.obs["domain"] == "domain_" + str(i)
                n_obs_domain = condition.sum()

                lambdas = domains_shift[i] + slide_shift + panel_shift
                X_domain = np.random.exponential(lambdas, size=(n_obs_domain, n_vars))
                adata.X[condition] = X_domain.astype(int)  # values should look like counts

            if n_drop:
                size = n_vars - n_drop - panel_index  # different number of genes
                var_indices = np.random.choice(n_vars, size=size, replace=False)
                adata = adata[:, var_indices].copy()

            adatas_panel.append(adata[: -1 - panel_index - slide_index].copy())  # different number of cells

        adata_panel = anndata.concat(adatas_panel)

        if compute_spatial_neighbors:
            spatial_neighbors(adata_panel, slide_key="slide_key")
            _drop_neighbors(adata_panel, index=3)  # ensure one node is not connected to any other

        adata_panel.layers["counts"] = adata_panel.X.copy()
        sc.pp.normalize_total(adata_panel)
        sc.pp.log1p(adata_panel)

        adatas.append(adata_panel)

    return adatas


def _drop_neighbors(adata: AnnData, index: int):
    for key in ["spatial_connectivities", "spatial_distances"]:
        adata.obsp[key] = adata.obsp[key].tolil()
        adata.obsp[key][index] = 0
        adata.obsp[key][:, index] = 0
        adata.obsp[key] = adata.obsp[key].tocsr()
        adata.obsp[key].eliminate_zeros()


def _read_h5ad_from_hub(name: str, row: pd.Series):
    from huggingface_hub import hf_hub_download

    file_path = f"{row['species']}/{row['tissue']}/{name}.h5ad"
    local_file = hf_hub_download(repo_id="MICS-Lab/novae", filename=file_path, repo_type="dataset")

    return sc.read_h5ad(local_file)


def load_dataset(
    pattern: str | None = None,
    tissue: list[str] | str | None = None,
    species: list[str] | str | None = None,
    technology: list[str] | str | None = None,
    custom_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    top_k: int | None = None,
    dry_run: bool = False,
) -> list[AnnData]:
    """Automatically load slides from the Novae dataset repository.

    !!! info "Selecting slides"
        The function arguments allow to filter the slides based on the tissue, species, and name pattern.
        Internally, the function reads [this dataset metadata file](https://huggingface.co/datasets/MICS-Lab/novae/blob/main/metadata.csv) to select the slides that match the provided filters.

    Args:
        pattern: Optional pattern to match the slides names.
        tissue: Optional tissue (or tissue list) to filter the slides. E.g., `"brain", "colon"`.
        species: Optional species (or species list) to filter the slides. E.g., `"human", "mouse"`.
        technology: Optional technology (or technology list) to filter the slides. E.g., `"xenium", or "visium_hd"`.
        custom_filter: Custom filter function that takes the metadata DataFrame (see above link) and returns a boolean Series to decide which rows should be kept.
        top_k: Optional number of slides to keep. If `None`, keeps all slides.
        dry_run: If `True`, the function will only return the metadata of slides that match the filters.

    Returns:
        A list of `AnnData` objects, each object corresponds to one slide.
    """
    metadata = pd.read_csv("hf://datasets/MICS-Lab/novae/metadata.csv", index_col=0)

    FILTER_COLUMN = [("species", species), ("tissue", tissue), ("technology", technology)]
    VALID_VALUES = {column: metadata[column].unique() for column, _ in FILTER_COLUMN}

    for column, value in FILTER_COLUMN:
        if value is not None:
            values = [value] if isinstance(value, str) else value
            valid_values = VALID_VALUES[column]

            assert all(
                value in valid_values for value in values
            ), f"Found invalid {column} value in {values}. Valid values are {valid_values}."

            metadata = metadata[metadata[column].isin(values)]

    if custom_filter is not None:
        metadata = metadata[custom_filter(metadata)]

    assert not metadata.empty, "No dataset found for the provided filters."

    if pattern is not None:
        where = metadata.index.str.match(pattern)
        assert len(where), f"No dataset found for the provided pattern ({', '.join(list(metadata.index))})."
        metadata = metadata[where]

    assert not metadata.empty, "No dataset found for the provided filters."

    if top_k is not None:
        metadata = metadata.head(top_k)

    if dry_run:
        return metadata

    log.info(f"Found {len(metadata)} h5ad file(s) matching the filters.")
    return [_read_h5ad_from_hub(name, row) for name, row in metadata.iterrows()]


def load_local_dataset(relative_path: str, files_black_list: list[str] = None) -> list[AnnData]:
    """Load one or multiple AnnData objects based on a relative path from the data directory

    Args:
        relative_path: Relative from from the data directory. If a directory, loads every .h5ad files inside it. Can also be a file, or a file pattern.

    Returns:
        A list of AnnData objects
    """
    data_dir = repository_root() / "data"
    full_path = data_dir / relative_path

    files_black_list = files_black_list or []

    if full_path.is_file():
        assert full_path.name not in files_black_list, f"File {full_path} is in the black list"
        log.info(f"Loading one adata: {full_path}")
        return [anndata.read_h5ad(full_path)]

    if ".h5ad" in relative_path:
        all_paths = list(data_dir.rglob(relative_path))
    else:
        all_paths = list(full_path.rglob("*.h5ad"))

    all_paths = [path for path in all_paths if path.name not in files_black_list]

    log.info(f"Loading {len(all_paths)} adata(s): {', '.join([str(path) for path in all_paths])}")
    return [anndata.read_h5ad(path) for path in all_paths]


def load_wandb_artifact(name: str) -> Path:
    import wandb

    api = wandb.Api()

    if not name.startswith("novae/"):
        name = f"novae/novae/{name}"

    artifact = api.artifact(name)

    artifact_path = wandb_log_dir() / "artifacts" / artifact.name

    if artifact_path.exists():
        log.info(f"Artifact {artifact_path} already downloaded")
    else:
        log.info(f"Downloading artifact at {artifact_path}")
        artifact.download(root=artifact_path)

    return artifact_path


GENE_NAMES_SUBSET = [
    "A2M",
    "ACKR1",
    "ACKR3",
    "ACTA2",
    "ADAM10",
    "ADAM17",
    "ADAM8",
    "AHR",
    "AKT1",
    "AKT2",
    "AKT3",
    "AMOTL2",
    "ANGPT1",
    "ANGPT2",
    "APC",
    "APOE",
    "AQP1",
    "ARAF",
    "ASCL2",
    "ATF3",
    "ATM",
    "ATR",
    "AURKB",
    "AXIN2",
    "BAK1",
    "BAX",
    "BCL2",
    "BCL2L1",
    "BCL6",
    "BIRC3",
    "BMI1",
    "BMP1",
    "BMP4",
    "BRAF",
    "BRCA1",
    "BRD4",
    "BST2",
    "BTLA",
    "C1QA",
    "C1QC",
    "CA4",
    "CADM1",
    "CASP8",
    "CCL17",
    "CCL2",
    "CCL21",
    "CCL22",
    "CCL26",
    "CCL28",
    "CCL3",
    "CCL4",
    "CCL5",
    "CCL8",
    "CCNB1",
    "CCND1",
    "CCR1",
    "CCR10",
    "CCR2",
    "CCR4",
    "CCR5",
    "CCR6",
    "CCR7",
    "CCR8",
    "CD101",
    "CD14",
    "CD163",
    "CD19",
    "CD1B",
    "CD1C",
    "CD1D",
    "CD1E",
    "CD2",
    "CD200",
    "CD207",
    "CD209",
    "CD22",
    "CD226",
    "CD24",
    "CD244",
    "CD248",
    "CD27",
    "CD274",
    "CD276",
    "CD28",
    "CD36",
    "CD37",
    "CD3D",
    "CD3E",
    "CD3G",
    "CD4",
    "CD40",
    "CD40LG",
    "CD44",
    "CD5",
    "CD52",
    "CD68",
    "CD69",
    "CD7",
    "CD70",
    "CD79A",
    "CD79B",
    "CD80",
    "CD83",
    "CD86",
    "CD8A",
    "CD8B",
    "CD9",
    "CDH1",
    "CDH5",
    "CDK2",
    "CDK4",
    "CDK6",
    "CDKN1A",
    "CDKN1B",
    "CEACAM1",
    "CEBPB",
    "CEBPD",
    "CHI3L1",
    "CHIT1",
    "CIITA",
    "CLDN5",
    "CLEC10A",
    "CLEC14A",
    "CLEC4A",
    "CLEC9A",
    "CLIC3",
    "CMKLR1",
    "COL4A1",
    "COL4A2",
    "COL5A1",
    "COL6A3",
    "COL8A1",
    "CPA3",
    "CR2",
    "CREBBP",
    "CSF1",
    "CSF1R",
    "CSF2RA",
    "CSF2RB",
    "CSF3R",
    "CTNNB1",
    "CTSG",
    "CX3CL1",
    "CX3CR1",
    "CXCL1",
    "CXCL10",
    "CXCL11",
    "CXCL12",
    "CXCL13",
    "CXCL16",
    "CXCL2",
    "CXCL8",
    "CXCL9",
    "CXCR1",
    "CXCR2",
    "CXCR3",
    "CXCR4",
    "CXCR5",
    "CXCR6",
    "CYBB",
    "DDIT3",
    "DIABLO",
    "DKK3",
    "DNMT1",
    "DNMT3A",
    "DUSP1",
    "DUSP6",
    "E2F1",
    "EGF",
    "EGFR",
    "EGR1",
    "ELANE",
    "ELN",
    "ENG",
    "EOMES",
    "EPCAM",
    "EPHB3",
    "EPHB4",
    "ERBB2",
    "ERBB3",
    "ETS1",
    "EZH2",
    "FAP",
    "FASLG",
    "FBLN5",
    "FCER1A",
    "FCER1G",
    "FCER2",
    "FCGR2A",
    "FCGR3A",
    "FCGR3B",
    "FGF1",
    "FGF2",
    "FGFBP2",
    "FGFR1",
    "FGFR2",
    "FGFR3",
    "FLI1",
    "FLT1",
    "FLT4",
    "FN1",
    "FOS",
    "FOXP3",
    "FSCN1",
    "FUT4",
    "FZD7",
    "GATA2",
    "GATA3",
    "GJB2",
    "GNLY",
    "GPX3",
    "GZMA",
    "GZMB",
    "GZMH",
    "GZMK",
    "HAVCR2",
    "HDAC1",
    "HDAC3",
    "HIF1A",
    "HLA-B",
    "HLA-C",
    "HLA-DMA",
    "HLA-DPA1",
    "HLA-DPB1",
    "HLA-DQA1",
    "HLA-DQB1",
    "HLA-DRA",
    "HLA-DRB1",
    "HRAS",
    "ICAM1",
    "ICAM2",
    "ICAM3",
    "ICOS",
    "ICOSLG",
    "IDH1",
    "IDO1",
    "IFIT3",
    "IFNAR1",
    "IFNAR2",
    "IFNG",
    "IFNGR1",
    "IFNGR2",
    "IGF1",
    "IGFBP3",
    "IGHG3",
    "IGHG4",
    "IKBKB",
    "IKZF2",
    "IKZF4",
    "IL10",
    "IL12A",
    "IL12B",
    "IL1B",
    "IL1R2",
    "IL23A",
    "IL2RA",
    "IL2RB",
    "IL32",
    "IL3RA",
    "IL4I1",
    "IL6",
    "IL6R",
    "IL7R",
    "INSR",
    "IRF3",
    "IRF4",
    "IRF5",
    "IRF7",
    "IRF8",
    "IRS1",
    "ITGA1",
    "ITGA4",
    "ITGA5",
    "ITGAE",
    "ITGAM",
    "ITGAX",
    "ITGB1",
    "ITGB2",
    "ITM2C",
    "JAK1",
    "JUN",
    "JUNB",
    "KCNQ1",
    "KDR",
    "KIT",
    "KITLG",
    "KLF2",
    "KLRB1",
    "KLRC1",
    "KLRF1",
    "KLRG1",
    "KLRK1",
    "KRAS",
    "LAG3",
    "LAMC2",
    "LAMP1",
    "LAMP3",
    "LCN2",
    "LDHA",
    "LEF1",
    "LGALS2",
    "LGALS9",
    "LHFPL6",
    "LIF",
    "LIPA",
    "LMNA",
    "LOX",
    "LRP1",
    "LRP5",
    "LRP6",
    "LYVE1",
    "LYZ",
    "MADCAM1",
    "MAFB",
    "MAML1",
    "MAP2K1",
    "MARCKS",
    "MARCO",
    "MCM2",
    "MCM6",
    "MFAP5",
    "MKI67",
    "MLH1",
    "MME",
    "MMP1",
    "MMP11",
    "MMP2",
    "MMP7",
    "MMP9",
    "MMRN2",
    "MRC1",
    "MS4A1",
    "MS4A6A",
    "MSH2",
    "MSH3",
    "MSH6",
    "MSR1",
    "MTOR",
    "MYBL2",
    "MYC",
    "MZB1",
    "NCAM1",
    "NDUFA4L2",
    "NEBL",
    "NEDD4",
    "NF1",
    "NFE2L2",
    "NFKB1",
    "NFKB2",
    "NFKBIA",
    "NKG7",
    "NLRC5",
    "NLRP3",
    "NMB",
    "NOS3",
    "NRAS",
    "NTAN1",
    "PCNA",
    "PDCD1",
    "PDCD1LG2",
    "PDGFA",
    "PDGFB",
    "PDGFC",
    "PDGFRA",
    "PDGFRB",
    "PDK1",
    "PDK4",
    "PDPN",
    "PECAM1",
    "PGF",
    "PIEZO1",
    "PIK3CA",
    "PIK3CG",
    "PKIB",
    "PLA2G2A",
    "PLAC8",
    "PLIN2",
    "PLK1",
    "PLOD2",
    "PLVAP",
    "POSTN",
    "PPARD",
    "PRAG1",
    "PRF1",
    "PTEN",
    "PTK2",
    "PTPRC",
    "PTTG1",
    "RAF1",
    "RB1",
    "RELA",
    "RELB",
    "RET",
    "RGMB",
    "RNASE6",
    "RORC",
    "S100A12",
    "S100A4",
    "S100A8",
    "S100A9",
    "SELL",
    "SELP",
    "SELPLG",
    "SERPINA1",
    "SERPINE1",
    "SERPINF1",
    "SHARPIN",
    "SIGLEC1",
    "SMAD2",
    "SMARCA4",
    "SMO",
    "SMOC2",
    "SNAI2",
    "SOCS3",
    "SOD2",
    "SOX2",
    "SOX9",
    "SPARCL1",
    "SPI1",
    "SPP1",
    "SPRY2",
    "SRC",
    "SRPRB",
    "STAT1",
    "STAT3",
    "STAT4",
    "STAT5A",
    "STAT6",
    "STING1",
    "STMN1",
    "TACSTD2",
    "TAGLN",
    "TAP2",
    "TAPBP",
    "TAT",
    "TBK1",
    "TBX21",
    "TBX3",
    "TCF4",
    "TCF7L2",
    "TEAD1",
    "TEAD4",
    "TEK",
    "TFF3",
    "TFPI",
    "TFPI2",
    "TGFB1",
    "TGFB2",
    "TGFB3",
    "TGFBI",
    "TGFBR1",
    "TGFBR2",
    "TGFBR3",
    "TGM2",
    "THBD",
    "TIGIT",
    "TIMP3",
    "TLR1",
    "TLR2",
    "TLR9",
    "TMEM37",
    "TMEM59",
    "TNF",
    "TNFRSF13B",
    "TNFRSF13C",
    "TNFRSF17",
    "TNFRSF18",
    "TNFRSF4",
    "TNFRSF9",
    "TNFSF10",
    "TNFSF4",
    "TNFSF9",
    "TOP2A",
    "TOX",
    "TP53",
    "TP63",
    "TPM1",
    "TPM2",
    "TRAC",
    "TRBC1",
    "TREM2",
    "TSPAN5",
    "TWIST1",
    "TYROBP",
    "UBE2C",
    "VCAM1",
    "VCAN",
    "VEGFA",
    "VEGFB",
    "VEGFC",
    "VSIR",
    "WDFY4",
    "WNT5A",
    "WWTR1",
    "XCL1",
    "XCR1",
    "YAP1",
    "ZAP70",
    "ZEB1",
    "ZNF683",
]
