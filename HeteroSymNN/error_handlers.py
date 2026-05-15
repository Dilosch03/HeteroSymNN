import functools
import inspect
from .config import settings

__all__ = ["clean_traceback", "apply_clean_tracebacks"]

def clean_traceback(func):
    """Method decorator to suppress deep tracebacks based on settings."""
    @functools.wraps(func)
    def wrapper_method(*args, **kwargs):
        if settings.debug_mode:
            return func(*args, **kwargs)
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_class = type(e)
            # Re-raise the exact same error type without the traceback stack
            raise error_class(f"{str(e)}") from None
            
    return wrapper_method

def apply_clean_tracebacks(cls):
    """Class decorator to apply @clean_traceback to all public methods."""
    for name, method in inspect.getmembers(cls, inspect.isfunction):
        # Wrap public methods and the constructor
        if not name.startswith('_') or name == '__init__':
            setattr(cls, name, clean_traceback(method))
    return cls