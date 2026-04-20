.. _configuration:

Global Settings & Configuration
===============================

The ``config`` module acts as the central nervous system for HeteroSymNN. It manages the global execution environment, handling hardware backend selection (CUDA vs. JIT CPU vs. Pure Python), thread allocation, disk caching mechanisms, and computational precision.

To ensure a perfectly synchronized state across all layers and wrappers, the configuration relies on a Singleton design pattern. Upon importing the library, an internal ``_Settings`` class is instantiated as the global ``settings`` object. 

During initialization, this object automatically evaluates your hardware environment and assigns the fastest available compute method:
1. **``GPU_CUDA``** (If a compatible NVIDIA GPU and CuPy are detected)
2. **``CPU_JIT``** (If a C++ compiler is available for Just-In-Time compilation)
3. **``CPU_PYTHON``** (Safe fallback to standard NumPy arrays)

When to Use
-----------
Because the framework automatically detects and utilizes the best hardware available, most users will never need to touch this module. You should only interact with the global settings if you need to:

* **Override Hardware:** Force the engine to use the CPU (for debugging or direct memory inspection) even if a GPU is present.
* **Manage Threads:** Restrict the number of CPU threads the framework is allowed to consume in constrained container environments.
* **Clear the Compiler Cache:** Purge the local disk of previously compiled C++/CUDA binaries to free up space or force a clean recompilation.

Code Example
------------
You can interact directly with the ``settings`` object or use the module-level helper functions to tailor HeteroSymNN to your current workload.

.. code-block:: python

    from HeteroSymNN import config

    # 1. Hardware Forcing
    # Force the engine to bypass the JIT compiler and use standard Python/NumPy
    config.settings.set_default_compute_method("CPU_PYTHON")

    # 2. Thread Management
    # Limit the framework to exactly 4 CPU threads
    config.settings.n_jobs = 4

    # 3. Cache Management
    # Clear all compiled C++ and CUDA binaries from the local disk cache
    config.clear_kernel_cache(cache_type="ALL")

API Reference
-------------

.. currentmodule:: HeteroSymNN.config

.. autodata:: settings
   :annotation: = Global instance of _Settings

   The active singleton managing all framework configurations.

.. autoclass:: _Settings
   :members:
   :undoc-members:
   :exclude-members: mapping