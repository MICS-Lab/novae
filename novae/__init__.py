import importlib.metadata
import logging

__version__ = importlib.metadata.version("novae")

from ._logging import configure_logger
from .model import Novae
from . import utils
from . import data

log = logging.getLogger("novae")
configure_logger(log)
