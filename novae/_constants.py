class Keys:
    # obs keys
    SWAV_CLASSES: str = "swav_classes"
    IS_VALID_OBS: str = "neighborhood_valid"
    SLIDE_ID: str = "slide_id"

    # obsm keys
    REPR: str = "representation"
    REPR_CORRECTED: str = "repr_corrected"

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
    ADATA_INDEX: str = "adata_index"
    N_BATCHES: str = "n_batches"


class Nums:
    EPS: float = 1e-8
    N_OBS_THRESHOLD: int = 100_000_000  # TODO: improve this using total RAM
    DELAUNAY_RADIUS_TH: int = 100
    MAX_GENES: int = 3_000
