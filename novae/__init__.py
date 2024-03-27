import importlib.metadata
import logging

from ._logging import configure_logger
from .model import Novae

__version__ = importlib.metadata.version("novae")

log = logging.getLogger("novae")
configure_logger(log)
