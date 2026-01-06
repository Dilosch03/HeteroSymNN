import numpy as np
import subprocess
import warnings

GPU_ENABLED = False
be = np
asnumpy = np.array 
NUM_GPUS = 0
cp = None
CPP_JIT_ENABLED = False

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
    warnings.warn("No c++ compatilbe compiler found. Training would be done using numpy")
    return False 

CPP_JIT_ENABLED = _check_cpp_compiler()

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



