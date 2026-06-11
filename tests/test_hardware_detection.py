import sys
import os
import pytest
import subprocess
import warnings
import importlib
from unittest.mock import patch, MagicMock

# Suppress numpy reload warnings which happen when we reload framework modules
warnings.filterwarnings("ignore", message=".*NumPy module was reloaded.*")
warnings.filterwarnings("ignore", message=".*There was a problem with the loading of CuPy.*")

# Ensure HeteroSymNN is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import exceptions safely
import HeteroSymNN.exceptions as exc

def reload_framework_modules():
    """
    Helper function to completely reload the hardware detection and config modules.
    This simulates a fresh import of the framework in a real system environment.
    """
    import HeteroSymNN.Backend.hardware as hw
    import HeteroSymNN.config as cfg
    
    importlib.reload(hw)
    
    # We patch Path.exists so that settings.json is not loaded from disk.
    # Otherwise, user's saved local config will overwrite the detected defaults.
    with patch('HeteroSymNN.config.Path.exists', return_value=False):
        importlib.reload(cfg)
    
    return hw, cfg

class TestHardwareDetection:
    """
    Test suite dedicated to testing agnostic hardware detection.
    This dynamically manipulates available system packages (like cupy) and subprocess outputs
    BEFORE loading the framework's modules to properly test the runtime detection logic.
    """

    def test_no_gpu_installed(self):
        """Test framework initialization when CuPy is completely missing from the system."""
        with patch.dict('sys.modules', {'cupy': None}):
            # We already filter the CuPy failure warning globally, but let's be sure it processes correctly.
            hw, cfg = reload_framework_modules()
            
            assert hw.GPU_ENABLED is False
            assert hw.NUM_GPUS == 0
            assert hw.cp is None
            
            assert "GPU_CUDA" not in cfg.settings.available_methods
            assert cfg.settings.default_compute_method == "CPU_PYTHON"

    def test_gpu_installed_successfully(self):
        """Test framework initialization when CuPy is available and detects GPUs."""
        class MockDevice:
            def __init__(self, device_id):
                self.attributes = {'MaxThreadsPerBlock': 512}

        class MockRuntime:
            @staticmethod
            def getDeviceCount():
                return 2

        class MockCuda:
            runtime = MockRuntime()
            Device = MockDevice

        class MockCuPy:
            cuda = MockCuda()
            @staticmethod
            def asnumpy(x): return x
        
        with patch.dict('sys.modules', {'cupy': MockCuPy}):
            hw, cfg = reload_framework_modules()
            
            assert hw.GPU_ENABLED is True, "GPU_ENABLED should be True"
            assert hw.NUM_GPUS == 2
            
            assert "GPU_CUDA" in cfg.settings.available_methods
            assert cfg.settings.default_compute_method == "GPU_CUDA"

    def test_cpp_compiler_not_found(self):
        """Test detection when no C++ compiler is available in the system PATH."""
        def mock_subprocess_run(*args, **kwargs):
            raise FileNotFoundError("Mock compiler not found")
            
        with patch('subprocess.run', side_effect=mock_subprocess_run):
            with pytest.warns(exc.HardwareWarning, match="No c\\+\\+ compatilbe compiler found"):
                hw, _ = reload_framework_modules()
                
            assert hw.CPP_JIT_ENABLED is False

    def test_cpp_compiler_found(self):
        """Test detection when a C++ compiler is available."""
        def mock_subprocess_run(*args, **kwargs):
            return MagicMock() # Simulates successful exit code 0
            
        with patch('subprocess.run', side_effect=mock_subprocess_run):
            with warnings.catch_warnings():
                warnings.simplefilter("error") # Ensure no warnings are raised (except ignored ones)
                hw, _ = reload_framework_modules()
                
            assert hw.CPP_JIT_ENABLED is True
            assert hw.CPP_INSTALLED_COMPILER in ["cl.exe", "g++", "clang"]

    def test_set_compute_method_fallback(self):
        """Test that manually asking for GPU_CUDA when unavailable falls back to CPU_PYTHON."""
        with patch.dict('sys.modules', {'cupy': None}):
            _, cfg = reload_framework_modules()
            
            with pytest.warns(exc.BackendNotAvailableWarning, match="no GPU is available"):
                cfg.settings.set_default_compute_method("GPU_CUDA")
                
            assert cfg.settings.default_compute_method == "CPU_PYTHON"

    def test_get_cuda_dims_gpu_disabled(self):
        """Test CUDA grid/block dimensions fallback when GPU is disabled at load time."""
        with patch.dict('sys.modules', {'cupy': None}):
            hw, _ = reload_framework_modules()
            
            grid_dim, block_dim = hw._get_cuda_dims(1000, 0)
            
            assert block_dim == (256,)
            assert grid_dim == ((1000 + 256 - 1) // 256,)
            
    def test_get_cuda_dims_gpu_enabled(self):
        """Test CUDA grid/block dimensions when GPU is available at load time."""
        class MockDevice:
            def __init__(self, device_id):
                self.attributes = {'MaxThreadsPerBlock': 512}
                
        class MockRuntime:
            @staticmethod
            def getDeviceCount(): return 1
            
        class MockCuda:
            Device = MockDevice
            runtime = MockRuntime()
                
        class MockCuPy:
            cuda = MockCuda()
            @staticmethod
            def asnumpy(x): return x
        
        with patch.dict('sys.modules', {'cupy': MockCuPy}):
            hw, _ = reload_framework_modules()
            
            grid_dim, block_dim = hw._get_cuda_dims(1000, 0)
            
            assert block_dim == (512,)
            assert grid_dim == ((1000 + 512 - 1) // 512,)
