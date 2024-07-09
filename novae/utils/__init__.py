from ._docs import format_docs
from ._build import spatial_neighbors
from .utils import (
    prepare_adatas,
    repository_root,
    tqdm,
    fill_invalid_indices,
    lower_var_names,
    wandb_log_dir,
    pretty_num_parameters,
    parse_device_args,
)
from ._plot import plot_niches_hierarchy
from ._data import _load_dataset, _load_wandb_artifact, dummy_dataset
