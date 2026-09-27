import functools
import inspect
from .config import settings
from .exceptions import HeteroSymNNError

__all__ = ["clean_traceback", "apply_clean_tracebacks"]

def clean_traceback(func):
    """
    Method decorator to suppress deep tracebacks based on settings.

    Only the framework's own errors (:exc:`~HeteroSymNN.exceptions.HeteroSymNNError`) are cleaned:
    they explain a user mistake, so the internal frames are noise. Any other exception is a bug and
    keeps its full traceback. With ``settings.debug_mode`` nothing is cleaned.
    """
    @functools.wraps(func)
    def wrapper_method(*args, **kwargs):
        if settings.debug_mode:
            return func(*args, **kwargs)

        try:
            return func(*args, **kwargs)
        except HeteroSymNNError as e:
            # Re-raise the SAME exception object, so its type, arguments and attributes are kept
            # whatever its constructor looks like. Only the internal frames and the chained
            # exceptions behind it are dropped.
            raise e.with_traceback(None) from None

    return wrapper_method

def apply_clean_tracebacks(cls):
    """Class decorator to apply @clean_traceback to all public methods."""
    for name, method in inspect.getmembers(cls, inspect.isfunction):
        # Wrap public methods and the constructor
        if not name.startswith('_') or name == '__init__':
            setattr(cls, name, clean_traceback(method))
    return cls