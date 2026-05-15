import numpy as np
import os
import subprocess
import warnings

from ..exceptions import HardwareWarning

__all__ = [
    "GPU_ENABLED", "NUM_GPUS", "CPP_JIT_ENABLED", "NUM_CPU_THREADS",
    "be", "cp", "asnumpy", "CPP_INSTALLED_COMPILER",
]

GPU_ENABLED = False
be = np
asnumpy = np.array 
NUM_GPUS = 0
cp = None
CPP_JIT_ENABLED = False
NUM_CPU_THREADS = os.cpu_count()
CPP_INSTALLED_COMPILER = None

try: 
    import cupy

    NUM_GPUS = cupy.cuda.runtime.getDeviceCount()
    if (NUM_GPUS>0):
        be = cupy
        cp = cupy
        GPU_ENABLED = True
        asnumpy = cp.asnumpy
except Exception as e:
    warnings.warn("There was a problem with the loading of CuPy and training will be done in the CPU.",HardwareWarning,stacklevel=2)

def _check_cpp_compiler()->bool:
    """
    Checks if there is a c++ compiler installed in the computer and can be access globaly.
    For internal use, recomended calling it just for debugging or logging purposes.
    
    Returns
    -------
    bool
        True if a c++ compiler is found, False otherwise.
    """
    compilers = [['cl.exe', '/?'],['g++', '--version'], ['clang', '--version']] 
    global CPP_INSTALLED_COMPILER
    for args in compilers:
        try:
            subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            CPP_INSTALLED_COMPILER = args[0]
            return True
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            continue
    warnings.warn("No c++ compatilbe compiler found. Training would be done using numpy",HardwareWarning,stacklevel=2)
    return False 

CPP_JIT_ENABLED = _check_cpp_compiler()

def _get_cuda_dims(n:int,device_id:int)->tuple[int, int]:
    """
    Not yet utilized internal method to get the number of dims a certain gpu has.

    Parameters
    ----------
    n: int
        size of the mange data
    device_id: int
        gpu id that is going to be used
    
    Returns
    -------
    tuple[int, int]
        tuple with the number size of the grid and the size of the block
    """
    max_threads = 256
    if(GPU_ENABLED):
        try:
            max_threads = cp.cuda.Device(device_id).attributes['MaxThreadsPerBlock']
        except Exception:
            pass

    block_dim = (max_threads,)
    grid_dim = ((n + block_dim[0] - 1) // block_dim[0],)
    return grid_dim, block_dim



