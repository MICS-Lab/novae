import warnings

from ._utils import (
    fill_invalid_indices,
    get_reference,
    iter_slides,
    lower_var_names,
    parse_device_args,
    pretty_model_repr,
    pretty_num_parameters,
    repository_root,
    tqdm,
    train,
    unique_leaves_indices,
    unique_obs,
    valid_indices,
    wandb_log_dir,
)
from ._validate import (
    check_available_domains_key,
    check_has_spatial_adjancency,
    check_slide_name_key,
    prepare_adatas,
    check_model_name,
)
from .build import spatial_neighbors
from .correct import batch_effect_correction
from .mode import Mode


def load_dataset(*args, **kwargs):
    warnings.warn(
        "The `novae.utils.load_dataset` function is deprecated and will be removed in `novae==1.1.0`. Please use `novae.load_dataset` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .. import load_dataset

    return load_dataset(*args, **kwargs)


def quantile_scaling(*args, **kwargs):
    warnings.warn(
        "The `novae.utils.quantile_scaling` function is deprecated and will be removed in `novae==1.1.0`. Please use `novae.quantile_scaling` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .. import quantile_scaling

    return quantile_scaling(*args, **kwargs)


def toy_dataset(*args, **kwargs):
    warnings.warn(
        "The `novae.utils.toy_dataset` function is deprecated and will be removed in `novae==1.1.0`. Please use `novae.toy_dataset` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .. import toy_dataset

    return toy_dataset(*args, **kwargs)
