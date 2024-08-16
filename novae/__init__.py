import importlib.metadata
import logging

__version__ = importlib.metadata.version("novae")

from ._logging import configure_logger
from .model import Novae
from . import utils
from . import data
from . import monitor
from . import plot
from ._constants import settings

log = logging.getLogger("novae")
configure_logger(log)
