Hardware & Backends
===================

The **Backends** module handles hardware detection, memory management, and caching.

Hardware Detection
------------------

.. automodule:: HeteroSymNN.Backends.hardware
   :members:
   :undoc-members:
   :show-inheritance:

   **Global Flags:**

   * ``GPU_ENABLED``: (bool) True if a CUDA-capable GPU and CuPy are detected.
   * ``CPP_JIT_ENABLED``: (bool) True if a C++ compiler (g++, clang, cl.exe) is found.
   * ``NUM_GPUS``: (int) The number of available GPUs.

Utilities
---------

.. autofunction:: HeteroSymNN.Backends.hardware.clear_kernel_cache

   Clears the compiled kernel cache to force recompilation.

   :param cache_type: Which cache to clear. Options: ``"ALL"``, ``"CPU"``, ``"GPU"``.
   
   .. warning::
      Clearing the GPU cache might require deleting files in the user's home directory, which may require permissions.