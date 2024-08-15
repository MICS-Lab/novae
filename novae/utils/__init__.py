from ._docs import format_docs
from ._build import spatial_neighbors
from ._utils import (
    prepare_adatas,
    repository_root,
    tqdm,
    fill_invalid_indices,
    lower_var_names,
    wandb_log_dir,
    pretty_num_parameters,
    pretty_model_repr,
    parse_device_args,
    requires_fit,
    get_valid_indices,
    unique_leaves_indices,
    unique_obs,
    _sparse_std,
    _check_available_obs_key,
    _set_unique_slide_ids,
)
from ._data import _load_dataset, _load_wandb_artifact, dummy_dataset
from ._mode import Mode
from ._correction import batch_effect_correction
