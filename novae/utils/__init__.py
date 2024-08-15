from ._docs import format_docs
from ._build import spatial_neighbors
from ._utils import (
    repository_root,
    tqdm,
    fill_invalid_indices,
    lower_var_names,
    wandb_log_dir,
    pretty_num_parameters,
    pretty_model_repr,
    parse_device_args,
    requires_fit,
    valid_indices,
    unique_leaves_indices,
    unique_obs,
    sparse_std,
)
from ._validate import check_available_domains_key, prepare_adatas
from ._data import load_dataset, load_wandb_artifact, dummy_dataset
from ._mode import Mode
from ._correct import batch_effect_correction
