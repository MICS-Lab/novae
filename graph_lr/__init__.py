import importlib.metadata
import logging
from ._logging import configure_logger

__version__ = importlib.metadata.version("graph_lr")

log = logging.getLogger("graph_lr")
configure_logger(log)

from .model import GraphLR
