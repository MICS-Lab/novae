import numpy as np


class Keys:
    # obs keys
    LEAVES: str = "novae_leaves"
    DOMAINS_PREFIX: str = "novae_domains_"
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
    DEFAULT_SAMPLE_CELLS: int = 100_000
    WARMUP_EPOCHS: int = 1

    # distances constants and thresholds (in microns)
    CELLS_CHARACTERISTIC_DISTANCE: int = 20  # characteristic distance between two cells, in microns
    DELAUNAY_RADIUS_TH: int = 100
    MEAN_DISTANCE_UPPER_TH_WARNING: float = 50
    MEAN_DISTANCE_LOWER_TH_WARNING: float = 4

    # genes constants
    N_HVG_THRESHOLD: int = 500
    MIN_GENES_FOR_HVG: int = 100
    MIN_GENES: int = 20

    # swav head constants
    SWAV_EPSILON: float = 0.05
    SINKHORN_ITERATIONS: int = 3
    QUEUE_SIZE: int = 5
    QUEUE_WEIGHT_THRESHOLD: float = 0.99

    # misc nums
    MEAN_NGH_TH_WARNING: float = 3.5
    N_OBS_THRESHOLD: int = 2_000_000  # above this number, lazy loading is used

    def disable_lazy_loading(self):
        """Disable lazy loading of subgraphs in the NovaeDataset."""
        Nums.N_OBS_THRESHOLD = np.inf

    def enable_lazy_loading(self, n_obs_threshold: int = 0):
        """Enable lazy loading of subgraphs in the NovaeDataset.

        Args:
            n_obs_threshold: Lazy loading is used above this number of cells in an AnnData object.
        """
        Nums.N_OBS_THRESHOLD = n_obs_threshold

    @property
    def warmup_epochs(self):
        return Nums.WARMUP_EPOCHS

    @warmup_epochs.setter
    def warmup_epochs(self, value: int):
        Nums.WARMUP_EPOCHS = value


settings = Nums()
