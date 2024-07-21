import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData

from . import repository_root, spatial_neighbors, wandb_log_dir

log = logging.getLogger(__name__)


def _load_dataset(relative_path: str) -> list[AnnData]:
    """Load one or multiple AnnData objects based on a relative path from the data directory

    Args:
        relative_path: Relative from from the data directory. If a directory, loads every .h5ad files inside it. Can also be a file, or a file pattern.

    Returns:
        A list of AnnData objects
    """
    data_dir = repository_root() / "data"
    full_path = data_dir / relative_path

    if full_path.is_file():
        log.info(f"Loading one adata: {full_path}")
        return [anndata.read_h5ad(full_path)]

    if ".h5ad" in relative_path:
        all_paths = list(map(str, data_dir.rglob(relative_path)))
    else:
        all_paths = list(map(str, full_path.rglob("*.h5ad")))

    log.info(f"Loading {len(all_paths)} adata(s): {', '.join(all_paths)}")
    return [anndata.read_h5ad(path) for path in all_paths]


def dummy_dataset(
    n_obs_per_domain: int = 1000,
    n_vars: int = 100,
    n_drop: int = 20,
    n_domains: int = 4,
    n_panels: int = 3,
    n_slides_per_panel: int = 1,
    panel_shift_factor: float = 0.5,
    batch_shift_factor: float = 0.2,
    class_shift_factor: float = 2,
    slide_ids_unique: bool = True,
    compute_spatial_neighbors: bool = True,
) -> list[AnnData]:
    """Creates a dummy dataset, useful for debugging or testing.

    Args:
        n_obs_per_domain: Number of obs per domain or niche.
        n_vars: Number of genes.
        n_drop: Number of genes that are removed for each `AnnData` object. It will create non-identical panels.
        n_domains: Number of domains, or niches.
        n_panels: Number of panels. Each panel will correspond to one output `AnnData` object.
        n_slides_per_panel: Number of slides per panel.
        panel_shift_factor: Shift factor for each panel.
        batch_shift_factor: Shift factor for each batch.
        class_shift_factor: Shift factor for each niche.
        slide_ids_unique: Whether to ensure that slide ids are unique.
        compute_spatial_neighbors: Whether to compute the spatial neighbors graph.

    Returns:
        A list of `AnnData` objects representing a valid `Novae` dataset.
    """
    assert n_obs_per_domain > 10
    assert n_vars - n_drop - n_panels > 2

    panels_shift = [panel_shift_factor * np.random.randn(n_vars) for _ in range(n_panels)]
    domains_shift = [class_shift_factor * np.random.randn(n_vars) for _ in range(n_domains)]
    loc_shift = [np.array([0, 10 * i]) for i in range(n_domains)]

    adatas = []

    var_names = np.array(
        TRUE_GENE_NAMES[:n_vars] if n_vars <= len(TRUE_GENE_NAMES) else [f"g{i}" for i in range(n_vars)]
    )

    for panel_index in range(n_panels):
        X_, spatial_, domains_, slide_ids_ = [], [], [], []

        slide_key = f"slide_{panel_index}_" if slide_ids_unique else "slide_"
        if n_slides_per_panel > 1:
            slides_shift = np.array([batch_shift_factor * np.random.randn(n_vars) for _ in range(n_slides_per_panel)])

        for domain_index in range(n_domains):
            n_obs = n_obs_per_domain + panel_index  # ensure n_obs different
            cell_shift = np.random.randn(n_obs, n_vars)
            slide_ids_domain_ = np.random.randint(0, n_slides_per_panel, n_obs)
            X_domain_ = cell_shift + domains_shift[domain_index] + panels_shift[panel_index]

            if n_slides_per_panel > 1:
                X_domain_ += slides_shift[slide_ids_domain_]

            X_.append(X_domain_)
            spatial_.append(np.random.randn(n_obs, 2) + loc_shift[domain_index])
            domains_.append(np.array([f"domain_{domain_index}"] * n_obs))
            slide_ids_.append(slide_ids_domain_)

        X = np.concatenate(X_, axis=0).clip(0)

        if n_drop > 0:
            size = n_vars - n_drop - panel_index  # ensure n_vars different
            var_indices = np.random.choice(n_vars, size=size, replace=False)
            X = X[:, var_indices]
            var_names_ = var_names[var_indices]

        adata = AnnData(X=X)

        adata.obs_names = [f"c_{panel_index}_{i}" for i in range(adata.n_obs)]
        adata.var_names = var_names_
        adata.obs["domain"] = np.concatenate(domains_)
        adata.obs["slide_key"] = (slide_key + pd.Series(np.concatenate(slide_ids_)).astype(str)).values
        adata.obsm["spatial"] = np.concatenate(spatial_, axis=0)

        adata.obsm["spatial"][[1, 2]] += 100  # ensure at least two indices are connected to only one node
        adata.obsm["spatial"][4] += 1000  # ensure at least one index is non-connected

        if compute_spatial_neighbors:
            spatial_neighbors(adata, radius=[0, 3])

        adatas.append(adata)

    return adatas


def _load_wandb_artifact(name: str) -> Path:
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


TRUE_GENE_NAMES = [
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
