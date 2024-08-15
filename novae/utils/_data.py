import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData

from . import repository_root, spatial_neighbors, wandb_log_dir

log = logging.getLogger(__name__)


def dummy_dataset(
    n_panels: int = 3,
    n_domains: int = 4,
    n_slides_per_panel: int = 1,
    xmax: int = 500,
    n_vars: int = 100,
    n_drop: int = 20,
    step: int = 20,
    panel_shift_lambda: float = 0.25,
    slide_shift_lambda: float = 0.5,
    domain_shift_lambda: float = 0.25,
    slide_ids_unique: bool = True,
    compute_spatial_neighbors: bool = False,
) -> list[AnnData]:
    """Creates a dummy dataset, useful for debugging or testing.

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

    domain = "domain_" + (np.sqrt((spatial**2).sum(1)) // (xmax / n_domains + 1e-8)).astype(int).astype(str)

    adatas = []

    var_names = np.array(
        TRUE_GENE_NAMES[:n_vars] if n_vars <= len(TRUE_GENE_NAMES) else [f"g{i}" for i in range(n_vars)]
    )

    lambdas_per_domain = np.random.exponential(1, size=(n_domains, n_vars))

    for panel_index in range(n_panels):
        adatas_panel = []
        panel_shift = np.random.exponential(panel_shift_lambda, size=n_vars)

        for slide_index in range(n_slides_per_panel):
            slide_shift = np.random.exponential(slide_shift_lambda, size=n_vars)

            adata = AnnData(
                np.zeros((n_obs, n_vars)),
                obsm={"spatial": spatial + panel_index + slide_index},  # ensure the locs are different
                obs=pd.DataFrame({"domain": domain}, index=[f"cell_{i}" for i in range(spatial.shape[0])]),
            )

            adata.var_names = var_names
            adata.obs_names = [f"c_{panel_index}_{slide_index}_{i}" for i in range(adata.n_obs)]

            slide_key = f"slide_{panel_index}_{slide_index}" if slide_ids_unique else f"slide_{slide_index}"
            adata.obs["slide_key"] = slide_key

            for i in range(n_domains):
                condition = adata.obs["domain"] == "domain_" + str(i)
                n_obs_domain = condition.sum()

                domain_shift = np.random.exponential(domain_shift_lambda, size=n_vars)
                lambdas = lambdas_per_domain[i] + domain_shift + slide_shift + panel_shift
                X_domain = np.random.exponential(lambdas, size=(n_obs_domain, n_vars))
                adata.X[condition] = X_domain.clip(0, 9)  # values should look like log1p values

            if n_drop:
                size = n_vars - n_drop - panel_index  # different number of genes
                var_indices = np.random.choice(n_vars, size=size, replace=False)
                adata = adata[:, var_indices].copy()

            adatas_panel.append(adata[: -1 - panel_index - slide_index].copy())  # different number of cells

        adata_panel = anndata.concat(adatas_panel)

        if compute_spatial_neighbors:
            spatial_neighbors(adata_panel, slide_key="slide_key")
            _drop_neighbors(adata_panel, index=3)  # ensure one node is not connected to any other

        adatas.append(adata_panel)

    return adatas


def _drop_neighbors(adata: AnnData, index: int):
    for key in ["spatial_connectivities", "spatial_distances"]:
        adata.obsp[key] = adata.obsp[key].tolil()
        adata.obsp[key][index] = 0
        adata.obsp[key][:, index] = 0
        adata.obsp[key] = adata.obsp[key].tocsr()
        adata.obsp[key].eliminate_zeros()


def load_dataset(relative_path: str) -> list[AnnData]:
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
