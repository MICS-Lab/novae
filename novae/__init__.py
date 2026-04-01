import importlib.metadata
import logging

__version__ = importlib.metadata.version("novae")

from ._logging import configure_logger
from ._settings import settings
from .model import Novae
from . import utils
from . import data
from . import monitor
from . import plot
from .utils import spatial_neighbors, batch_effect_correction
from .data import load_dataset, toy_dataset, quantile_scaling, compute_histo_embeddings, compute_histo_pca

log = logging.getLogger("novae")
configure_logger(log)
