from platformdirs import user_cache_dir
from pathlib import Path
from typing import Literal
import os
import shutil
import warnings
import numpy as np

from .Backend.hardware import GPU_ENABLED,CPP_JIT_ENABLED

USE_KERNEL_CACHE =True
CPU_CACHE_DIR = Path(user_cache_dir("HeteroSymNN"),"CPU")
KERNEL_CACHE = {}
WARNINGS_STRICT_MODE = False

DEFAULT_COMPUTE_METHOD = "CPU_PYTHON"
if GPU_ENABLED:
    DEFAULT_COMPUTE_METHOD = "GPU_CUDA"
elif CPP_JIT_ENABLED:
    DEFAULT_COMPUTE_METHOD = "CPU_CPP"

DEFAULT_DTYPE = np.float32

def clear_kernel_cache(cache_type: Literal["ALL","CPU","GPU"] = 'ALL')->None:
    """
    Clears the cache used by HeteroSymNN.
    
    Parameters
    ----------
    cache_type : Literal["ALL","CPU","GPU"], optional
        Type of cache to clear. Options are:
        - "ALL": Clears both CPU and GPU caches.
        - "CPU": Clears only the CPU cache.
        - "GPU": Clears only the GPU cache.
        The default is 'ALL'.
    """
    cache_type = cache_type.upper()
    
    def remove_dir(dir_path, name):
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                warnings.warn(f"Error al eliminar el caché {name}: {e}")

    if (cache_type in ('ALL', 'CPU')):
        remove_dir(CPU_CACHE_DIR, "CPU")
        

    if (cache_type in ('ALL', 'GPU')):
        try:
            home_dir = os.path.expanduser('~')
            cupy_cache_dir = os.path.join(home_dir, '.cupy', 'kernel_cache')
            if os.path.exists(cupy_cache_dir):
                warnings.warn(f"Modulo Custom_AI no tiene permisos para eliminar los caches de cupy. Ubicacion de los caches de cupy: {cupy_cache_dir} si realmente quere eliminarlos hacerlos a su discreción.")

        except Exception as e:
            warnings.warn(f"No se pudo localizar el directorio de caché de CuPy: {e}", RuntimeWarning)


    if (cache_type in ('ALL', 'CPU', 'GPU')):
        try:
            KERNEL_CACHE.clear()
            print("Caché de memoria (KERNEL_CACHE) limpiado.")
        except Exception as e:
            warnings.warn(f"No se pudo limpiar el caché de memoria: {e}")

def set_precision(dtype_str: str):
    """
    Sets the global data type for the engine.
    Supports standard floating point and integer types for future quantization.
    """
    global DEFAULT_DTYPE
    
    mapping = {
        "float32": np.float32,
        "float64": np.float64,
        "float16": np.float16,
        "int8": np.int8,
        "uint8": np.uint8
    }
    
    if dtype_str in mapping:
        DEFAULT_DTYPE = mapping[dtype_str]
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(mapping.keys())}")