.. _configuration:

Configuration
==============

The ``config`` module acts as the central nervous system for HeteroSymNN. It manages the global execution environment, handling hardware backend selection, thread allocation, caching mechanisms, and computational precision.

Architecture & Initialization
-----------------------------

To ensure consistent state across the library, the configuration relies on a Singleton design pattern. Upon importing the library, an internal ``_Settings`` class is instantiated as the ``settings`` object. 

During initialization, this object automatically evaluates the hardware environment and assigns the ``default_compute_method`` based on highest availability:
1. **``GPU_CUDA``** (If a compatible GPU and CuPy are detected)
2. **``CPU_JIT``** (If a C++ compiler is available for Just-In-Time compilation)
3. **``CPU_PYTHON``** (Fallback standard numpy implementation)

Usage Examples
--------------

You can interact directly with the ``settings`` object or use the module-level helper functions to tailor HeteroSymNN to your current workload.

.. code-block:: python

    from HeteroSymNN import config

    # 1. Thread Management
    # Limit the number of CPU threads (useful for constrained environments)
    config.settings.n_jobs = 4

    # 2. Backend Forcing
    # Force the engine to use a specific compute backend.
    # Note: If you request GPU_CUDA and no GPU is found, it will warn 
    # and safely fall back to the next available CPU method.
    config.settings.set_default_compute_method("CPU_JIT")

    # 3. Precision Control
    # Set global precision (e.g., for lower memory footprint or future quantization)
    config.set_precision("float16")

    # 4. Cache Management
    # Clear both CPU and GPU compiled kernel caches
    config.clear_kernel_cache(cache_type="ALL")

API Reference
-------------

.. currentmodule:: HeteroSymNN.config

The Settings Object
~~~~~~~~~~~~~~~~~~~

.. autodata:: settings
   :annotation: = Global instance of _Settings

.. autoclass:: _Settings
   :members:
   :undoc-members:
   :exclude-members: mapping

Module-Level Functions
~~~~~~~~~~~~~~~~~~~~~~

For convenience, the most frequently used setting methods are exposed directly at the module level.

.. autofunction:: clear_kernel_cache

.. autofunction:: set_precision