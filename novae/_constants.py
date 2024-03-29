# general
EPS = 1e-8
N_OBS_THRESHOLD = 1_000_000
DELAUNAY_RADIUS_TH = 100

# anndata general keys
COUNTS_LAYER = "counts"
VAR_MEAN = "mean"
VAR_STD = "std"
IS_KNOWN_GENE_KEY = "in_vocabulary"

# anndata neighborhood keys
IS_VALID_KEY = "neighborhood_valid"
ADJ = "spatial_distances"
ADJ_LOCAL = "spatial_distances_local"
ADJ_PAIR = "spatial_distances_pair"

# torch dataset keys
SLIDE_KEY = "slide_id"
N_BATCHES = "n_batches"
ADATA_INDEX_KEY = "adata_index"

# output keys
CODES = "codes"
SCORES = "adjacency_scores"
SWAV_CLASSES = "swav_classes"
REPR = "representation"
INT_CONF = "interaction_confidence"
