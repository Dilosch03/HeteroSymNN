
import numpy as np
import subprocess
import warnings
import os
import shutil
from typing import Literal

GPU_ENABLED = False
be = np
asnumpy = np.array 
NUM_GPUS = 0
cp = None
CPP_JIT_ENABLED = False
WARNINGS_STRICT_MODE = False
USE_KERNEL_CACHE =True
CPU_CACHE_DIR = os.path.join("kernel_cache", "cpu")
KERNEL_CACHE = {}

try: 
    import cupy

    NUM_GPUS = cupy.cuda.runtime.getDeviceCount()
    if (NUM_GPUS>0):
        be = cupy
        cp = cupy
        GPU_ENABLED = True
        asnumpy = cp.asnumpy
except Exception:
    warnings.warn("Cupy not installed. Training would be done in the CPU")

def _check_cpp_compiler():
    compilers = [['cl.exe', '/?'],['g++', '--version'], ['clang', '--version']] 
    global CPP_INSTALLED_COMPILER
    for args in compilers:
        try:
            subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            CPP_INSTALLED_COMPILER = args[0]
            return True
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            continue
    warnings.warn("No c++ compatilbe compiler found. Training would be done using numpy")
    return False 

CPP_JIT_ENABLED = _check_cpp_compiler()

DEFAULT_COMPUTE_METHOD = "CPU_PYTHON"
if GPU_ENABLED:
    DEFAULT_COMPUTE_METHOD = "GPU_CUDA"
elif CPP_JIT_ENABLED:
    DEFAULT_COMPUTE_METHOD = "CPU_CPP"

def _get_cuda_dims(n,device_id:int):
    max_threads = 256
    if(GPU_ENABLED):
        try:
            max_threads = cp.cuda.Device(device_id).attributes['MaxThreadsPerBlock']
        except Exception:
            pass

    block_dim = (max_threads,)
    grid_dim = ((n + block_dim[0] - 1) // block_dim[0],)
    return grid_dim, block_dim


def clear_kernel_cache(cache_type: Literal["ALL","CPU","GPU"] = 'ALL'):   
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
