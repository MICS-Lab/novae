class Keys:
    # obs keys
    SWAV_CLASSES: str = "novae_leaves"
    NICHE_PREFIX: str = "novae_niche_"
    IS_VALID_OBS: str = "neighborhood_valid"
    SLIDE_ID: str = "novae_sid"

    # obsm keys
    REPR: str = "novae_latent"
    REPR_CORRECTED: str = "novae_latent_corrected"

    # obsp keys
    ADJ: str = "spatial_distances"
    ADJ_LOCAL: str = "spatial_distances_local"
    ADJ_PAIR: str = "spatial_distances_pair"

    # var keys
    VAR_MEAN: str = "mean"
    VAR_STD: str = "std"
    IS_KNOWN_GENE: str = "in_vocabulary"
    HIGHLY_VARIABLE: str = "highly_variable"
    USE_GENE: str = "novae_use_gene"

    # layer keys
    COUNTS_LAYER: str = "counts"

    # misc keys
    UNS_TISSUE: str = "novae_tissue"
    ADATA_INDEX: str = "adata_index"
    N_BATCHES: str = "n_batches"
    NOVAE_VERSION: str = "novae_version"


class Nums:
    # training constants
    EPS: float = 1e-8
    MIN_DATASET_LENGTH: int = 50_000
    MAX_DATASET_LENGTH_RATIO: float = 0.02
    WARMUP_EPOCHS: int = 3

    # distances constants and thresholds (in microns)
    CELLS_CHARACTERISTIC_DISTANCE: int = 20  # characteristic distance between two cells, in microns
    DELAUNAY_RADIUS_TH: int = 100
    MEAN_DISTANCE_UPPER_TH_WARNING: float = 50
    MEAN_DISTANCE_LOWER_TH_WARNING: float = 4

    # genes constants
    MAX_GENES: int = 3_000
    MIN_GENES: int = 20

    # swav head constants
    SWAV_EPSILON: float = 0.05
    SINKHORN_ITERATIONS: int = 3
    QUEUE_SIZE: int = 5
    MIN_PROTOTYPES_RATIO: float = 0.2

    # misc nums
    MEAN_NGH_TH_WARNING: float = 3.5
    N_OBS_THRESHOLD: int = 2_000_000  # above this number, lazy loading is used
