from . import API, Core, JIT, Backend
from .config import settings,clear_kernel_cache,set_precision
from .exceptions import *

__all__ = ["API","Core","JIT","Backend","setTings","clear_kernel_cache","set_precision"]
__version__ = "1.0.0"