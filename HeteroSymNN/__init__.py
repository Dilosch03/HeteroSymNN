from . import API, Core, JIT, Backend, exceptions
from .config import settings,clear_kernel_cache,set_precision


__all__ = ["API","Core","JIT","Backend","exceptions","settings","clear_kernel_cache","set_precision"]
__version__ = "0.3.0"