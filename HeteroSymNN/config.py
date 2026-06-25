from platformdirs import user_cache_dir,user_config_dir
from pathlib import Path
from typing import Literal, Union
import os
import shutil
import warnings
import numpy as np
import json

from .Backend.hardware import GPU_ENABLED
from .exceptions import PathWarning,PathError,BackendNotAvailableWarning,HeteroSymNNWarnings,PerformanceWarning,ConfigError,ConfigWarning,ComputationalMethodValueError,DataTypeError, HeteroSymNNValueError
from .Backend import hardware as HW

__all__ = ["settings"]

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
        self._config_dir = Path(user_config_dir("HeteroSymNN"))
        self._config_file = self._config_dir / "settings.json"

        self.debug_mode = False
        self._num_cpu_threads = os.cpu_count()
        self._warning_level = "warn"
        self._use_kernel_cache = True
        self._cpu_cache_dir = Path(user_cache_dir("HeteroSymNN_Cache"),"CPU")
        self._kernel_cache = {}
        self._default_compute_method = "CPU_PYTHON"
        self._available_methods = ["CPU_PYTHON"]
        if GPU_ENABLED:
            self._default_compute_method = "GPU_CUDA"
            self._available_methods.append("GPU_CUDA")
        
        #elif CPP_JIT_ENABLED:
        #    self._default_compute_method = "CPU_JIT"
        
        if (self._default_compute_method == "GPU_CUDA"):
            self._default_manager = HW.cp
            self._default_asnumpy = HW.cp.asnumpy
        else:
            self._default_manager = np
            self._default_asnumpy = np.array

        self._warning_level = "default"
        warnings.simplefilter("default",HeteroSymNNWarnings)      
        self._default_dtype = np.float32

        self._load_from_file()

    @property
    def use_kernel_cache(self) -> bool:
        """
        Boolean atribute if the libary is going to cache the created kernels. Default is True.
        """
        return self._use_kernel_cache
    
    @property
    def available_methods(self) -> list[str]:
        """
        list of the available compuational methods that the library can use.
        """
        return self._available_methods

    @use_kernel_cache.setter
    def use_kernel_cache(self, value: bool):
        self._use_kernel_cache = bool(value)

    @property
    def n_jobs(self) -> int:
        """
        Number of threads that the library is allow to use. Default is all the available threads.
        """
        return self._num_cpu_threads
    
    @n_jobs.setter
    def n_jobs(self, value: int):
        if (value > os.cpu_count()):
            warnings.warn("Tried to use more threads than available. Using all available threads",PerformanceWarning,stacklevel=3)
        if (value == -1):
            value = os.cpu_count()
        if (value < 1):
            raise HeteroSymNNValueError("Thread value can't be less than 1, unless is -1 to signify all available threads.")
        self._num_cpu_threads = value

    
    @property
    def warning_level(self) -> str:
        """
        Level of how strict the library is with warnings.
        
        Options are:
            *"ignore"*
            *"always"*
            *"default"*
            *"module"*
            *"once"*
            *"error"*

        Default is "default".
        """
        return self._warning_level

    def set_warning_level(self, value: Literal["error", "ignore", "always", "default", "module","once"]):
        """
        Sets how the framework treats warnings.

        Args:
            value (Literal["error", "ignore", "always", "default", "module","once"]):
                "error" - Treat all warnings as errors
                "ignore" - Ignore all warnings
                "always" - Always show warnings
                "default" - Show warnings once
                "module" - Show warnings once per module
                "once" - Show warnings once

        Raises:
            ConfigError: If the warning level is invalid.
        """
        value = value.lower()
        if (value in ["error", "ignore", "always", "default", "module","once"]):
            self._warning_level = value
            warnings.simplefilter(value,HeteroSymNNWarnings)   
        else:
            raise ConfigError(f"Invalid warning level: {value}. Must be one of {['error', 'ignore', 'always', 'default', 'module','once']}")
        
    @property
    def cpu_cache_dir(self) -> Path:
        """
        Path to the directory used for storing JIT-compiled CPU kernels.
        """
        return self._cpu_cache_dir
    
    @property
    def kernel_cache(self) -> dict:
        """
        Ram cache currently used by HeteroSymNN.
        """
        return self._kernel_cache

    @property
    def default_compute_method(self) -> str:
        """
        Default compute method used by HeteroSymNN.

        Options are:
            *"GPU_CUDA"*
            *"CPU_JIT"*
            *"CPU_PYTHON"*
            
        Defaults to the hightest available method.

        If want to change the method use :meth:`set_default_compute_method`.
        """
        return self._default_compute_method
    
    @property
    def default_manager(self):
        """
        Default manager used by HeteroSymNN.
        """
        return self._default_manager
    
    @property
    def default_asnumpy(self):
        """
        Default method used for the asnumpy operations.
        """
        return self._default_asnumpy

    def _load_from_file(self):
        """Internal method to apply JSON settings over the defaults."""
        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    user_data = json.load(f)
                    
                    if "debug_mode" in user_data:
                        self.debug_mode = user_data["debug_mode"]
                    if "use_kernel_cache" in user_data:
                        self._use_kernel_cache = user_data["use_kernel_cache"]
                    if "warning_level" in user_data:
                        self.set_warning_level(user_data["warning_level"])
                    if "n_jobs" in user_data:
                        self.n_jobs = user_data["n_jobs"]
                    if "default_compute_method" in user_data:
                        self.set_default_compute_method(user_data["default_compute_method"])
            except Exception as e:
                warnings.warn(f"Failed to read settings.json. Corrupted or invalid format. Creating a new one with default settings. Error: {e}", ConfigWarning)
                self.save()

    def save(self):
        """Saves the current state of the settings object to the JSON file to act as the default settings for any project and future sessions. Creates a new file if it doesn't exist."""
        data = {
            "debug_mode": self.debug_mode,
            "use_kernel_cache": self._use_kernel_cache,
            "warning_level": self._warning_level,
            "n_jobs": self._num_cpu_threads,
            "default_compute_method": self._default_compute_method
        }
        
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            with open(self._config_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            warnings.warn(f"Failed to save settings.json: {e}", PathWarning)

    def set_default_compute_method(self, method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]):
        """
        Sets the default compute method used by HeteroSymNN.

        Parameters
        ----------
        method : Literal["GPU_CUDA", "CPU_JIT", "CPU_PYTHON"]
            The method to set as default.

        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.ComputationalMethodValueError`
            If the method is not one of the available methods.
        :exc:`~HeteroSymNN.exceptions.BackendNotAvailableWarning`
            If the method is not available on the current system.
        """
        new_method = method.upper()
        try_method = new_method
        msg_extra = ""
        if not(new_method in ["GPU_CUDA","CPU_JIT","CPU_PYTHON"]):
            raise ComputationalMethodValueError("tried to change the computational method to something that isn't GPU_CUDA, CPU_JIT or CPU_PYTHON")

        if ((new_method == "GPU_CUDA") and not(new_method in self.available_methods)):
            msg_extra = ", but no GPU is available."
            new_method = "CPU_PYTHON"

        if ((new_method == "CPU_JIT")):
                msg_extra = ", but currently is not available"
                new_method = "CPU_PYTHON"

        if (try_method != new_method):
                warnings.warn(f"Tried to change to use '{try_method}'{msg_extra}. {new_method} is required",BackendNotAvailableWarning,stacklevel=2)

        self._default_compute_method = new_method
        if (new_method == "GPU_CUDA"):
            self._default_manager = HW.cp
            self._default_asnumpy = HW.cp.asnumpy
        else:
            self._default_manager = np
            self._default_asnumpy = np.array

    @property
    def default_dtype(self)->np.dtype:
        """
        Default data type used by HeteroSymNN.
        Default used is float32.
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

        Currently the framework doesnt have the ability to change from float32.
        """
        return
        
        if dtype_str in self.mapping:
            self._default_dtype = self.mapping[dtype_str]
        else:
            raise DataTypeError(f"Unsupported dtype: {dtype_str}. Supported: {list(self.mapping.keys())}")
        
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
        :exc:`~HeteroSymNN.exceptions.PathError`
            If the path is the filesystem root. For security reasons it can not be set as the parent of the new cache directory.
            If the directory cannot be created or accessed.
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
                    self._cpu_cache_dir = new_cache_dir
                    
                except Exception as e:
                    warnings.warn(f"Failed to move some cache files: {e}. New cache is active but might be empty.",PathWarning,stacklevel=2)
        
settings = _Settings()