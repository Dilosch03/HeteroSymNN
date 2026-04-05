from platformdirs import user_cache_dir
from pathlib import Path
from typing import Literal, Union
import os
import shutil
import warnings
import numpy as np

from .Backend.hardware import GPU_ENABLED,CPP_JIT_ENABLED
from .exceptions import BackendNotAvailableError,PerformanceWarning,PathWarning,PathError,HardwareWarning
from .Backend import hardware as HW

class _Settings:
    
    """
    Global settings for HeteroSymNN.
    """
    mapping = {
            "float32": np.float32,
            "float64": np.float64,
            "float16": np.float16,
            "int8": np.int8,
            "uint8": np.uint8
        }
    def __init__(self):
        self._num_cpu_threads = os.cpu_count()
        self._warning_level = "warn"
        self._use_kernel_cache = True
        self._cpu_cache_dir = Path(user_cache_dir("HeteroSymNN_Cache"),"CPU")
        self._kernel_cache = {}
        self._default_compute_method = "CPU_PYTHON"
        if GPU_ENABLED:
            self._default_compute_method = "GPU_CUDA"
        elif CPP_JIT_ENABLED:
            self._default_compute_method = "CPU_JIT"
        
        if (self._default_compute_method == "GPU_CUDA"):
            self._default_manager = HW.cp
            self._default_asnumpy = HW.cp.asnumpy
        else:
            self._default_manager = np
            self._default_asnumpy = np.array
        
        
        self._default_dtype = np.float32

    @property
    def use_kernel_cache(self) -> bool:
        """
        """
        return self._use_kernel_cache

    @use_kernel_cache.setter
    def use_kernel_cache(self, value: bool):
        self._use_kernel_cache = bool(value)

    @property
    def n_jobs(self) -> int:
        """
        """
        return self._num_cpu_threads
    
    n_jobs.setter
    def n_jobs(self, value: int):
        if (value > os.cpu_count()):
            if (self._warning_level == "error"):
                raise ValueError
            elif (self._warning_level == "warn"):
                warnings.warn()
        self._num_cpu_threads = value

    
    @property
    def warning_level(self) -> str:
        """
        """
        return self._warning_level

    def set_warning_level(self, value: Literal["ignore","warn","error"]):
        value = value.lower()
        if (value in ["ignore","warn","error"]):
            self._warning_level = value
        else:
            raise ValueError(f"Invalid warning level: {value}. Must be one of {["ignore","warn","error"]}")
        
    @property
    def cpu_cache_dir(self) -> Path:
        """
        """
        return self._cpu_cache_dir
    
    @property
    def kernel_cache(self) -> dict:
        """
        """
        return self._kernel_cache

    @property
    def default_compute_method(self) -> str:
        """
        """
        return self._default_compute_method
    
    @property
    def default_manager(self):
        """
        """
        return self._default_manager
    
    @property
    def default_asnumpy(self):
        """
        """
        return self._default_asnumpy

    def set_default_compute_method(self, method: str):
        method = method.upper()
        valid_methods = ["GPU_CUDA", "CPU_JIT", "CPU_PYTHON"]
        
        if method not in valid_methods:
            raise ValueError(f"Invalid compute method: {method}. Must be one of {valid_methods}")

        if method == "GPU_CUDA" and not GPU_ENABLED:
            msg = "GPU_CUDA requested but GPU is not enabled."
            if self._warnings_strict_mode:
                raise BackendNotAvailableError(msg)
            warnings.warn(f"{msg} Falling back to CPU.",PerformanceWarning,stacklevel=2)
            method = "CPU_JIT" if CPP_JIT_ENABLED else "CPU_PYTHON"

        if method == "CPU_JIT" and not CPP_JIT_ENABLED:
            msg = "CPU_JIT requested but C++ compiler is not available."
            if self._warnings_strict_mode:
                raise BackendNotAvailableError(msg)
            warnings.warn(f"{msg} Falling back to CPU_PYTHON.",PerformanceWarning,stacklevel=2)
            method = "CPU_PYTHON"
            
        self._default_compute_method = method
        if (method == "GPU_CUDA"):
            self._default_manager = HW.cp
            self._default_asnumpy = HW.cp.asnumpy
        else:
            self._default_manager = np
            self._default_asnumpy = np.array

    @property
    def default_dtype(self)->np.dtype:
        """
        """
        return self._default_dtype

    def clear_kernel_cache(self,cache_type: Literal["ALL","CPU","GPU"] = 'ALL')->None:
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
                    warnings.warn(f"Error when clearing the cache{name}: {e}",PathWarning,stacklevel=2)

        if (cache_type in ('ALL', 'CPU')):
            remove_dir(self.cpu_cache_dir, "CPU")
            

        if (cache_type in ('ALL', 'GPU')):
            try:
                home_dir = os.path.expanduser('~')
                cupy_cache_dir = os.path.join(home_dir, '.cupy', 'kernel_cache')
                if os.path.exists(cupy_cache_dir):
                    warnings.warn(f"HeteroSymNN doen't have the permissions for clear the cupy cache. If really want to clear them the path is: {cupy_cache_dir}",PathWarning)
            except Exception as e:
                warnings.warn(f"Couldn't found the cupy cache path:{e}", PathWarning, stacklevel=2)


        if (cache_type in ('ALL', 'CPU', 'GPU')):
            try:
                self._kernel_cache.clear()
            except Exception as e:
                warnings.warn(f"Couldn't clear the ram cache: {e}",PathWarning,stacklevel=2)

    def set_precision(self,dtype_str: str)->None:
        """
        Sets the global data type for the engine.
        Supports standard floating point and integer types for future quantization.
        """
        
        if dtype_str in self.mapping:
            self._default_dtype = self.mapping[dtype_str]
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(self.mapping.keys())}")
        
    def set_cache_location(self,path: Union[str,Path],move_existing_cache: bool = False)->None:
        """
        Updates the directory used for storing JIT-compiled CPU kernels.
        
        Parameters
        ----------
        new_path : str | Path
            The base directory path. A 'HeteroSymNN_Cache' subdirectory will be created inside.
            If the path does not exist, it will be created.
        move_existing : bool, optional
            If True, moves files from the old cache directory to the new one.
            If False (default), the old cache is left as-is and the library starts fresh in the new location.
            
        Raises
        ------
        OSError
            If the directory cannot be created or accessed.
        ValueError
            If the path is the filesystem root. For security reasons it can not be set as the parent of the new cache directory.
        """
        if isinstance(path,str):
            path = Path(path).resolve()
        else:
            path = path.resolve()

        if (path.parent == path):
            raise PathError("Security Risk: Cannot set cache base to filesystem root.")
        
        new_cache_dir = path / "HeteroSymNN_Cache" / "CPU"
        old_cache_dir = self._cpu_cache_dir
        if (old_cache_dir != path):
            try:
                new_cache_dir.mkdir(parents=True,exist_ok=True)
            except OSError as e:
                raise PathError(f"Failed to create/access cache directory at {new_cache_dir}. Check permissions.") from e
            
            if  (move_existing_cache):
                try:
                    for item in os.listdir(old_cache_dir):
                        s = old_cache_dir / item
                        d = new_cache_dir / item
                        if s.is_file():
                            shutil.copy2(s, d)
                            os.remove(s)
                        elif s.is_dir():
                            shutil.copytree(s, d, dirs_exist_ok=True)
                            shutil.rmtree(s)
                    
                    try:
                        os.rmdir(old_cache_dir)
                    except OSError: pass 
                    
                except Exception as e:
                    warnings.warn(f"Failed to move some cache files: {e}. New cache is active but might be empty.",PathWarning,stacklevel=2)

        
        
settings = _Settings()

clear_kernel_cache = settings.clear_kernel_cache
set_precision = settings.set_precision
