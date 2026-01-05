Hardware & Backends
===================

The **Backends** module handles hardware detection, memory management, and caching.

.. module:: HeteroSymNN.Backend.hardware

Hardware Detection
------------------

These flags are set automatically upon import based on your system's capabilities.

.. data:: GPU_ENABLED
   :annotation: = bool

   ``True`` if a CUDA-capable GPU is detected and CuPy is installed.

.. data:: CPP_JIT_ENABLED
   :annotation: = bool

   ``True`` if a compatible C++ compiler (g++, clang, or cl.exe) is found in the system PATH.

.. data:: NUM_GPUS
   :annotation: = int

   The number of available CUDA devices.

Configuration Flags
-------------------

These variables control the framework's behavior and can be modified by the user at runtime.

.. data:: USE_KERNEL_CACHE
   :annotation: = bool (Default: True)

   If ``True``, compiled JIT kernels are saved to disk/memory to speed up future runs.
   Set to ``False`` to force re-compilation on every run (useful for debugging compiler changes).

.. data:: WARNINGS_STRICT_MODE
   :annotation: = bool (Default: False)

   If ``True``, the framework raises errors instead of warnings when hardware fallbacks occur (e.g., trying to use GPU mode without a GPU).

Internal State
--------------

These variables are for debugging and inspection purposes. They show how the framework has configured itself.

.. data:: DEFAULT_COMPUTE_METHOD
   :annotation: = str

   The default backend selected based on available hardware.
   
   * ``"GPU_CUDA"``: Best performance.
   * ``"CPU_CPP"``: Good performance (OpenMP accelerated).
   * ``"CPU_PYTHON"``: Slowest (Fallback).

.. data:: CPP_INSTALLED_COMPILER
   :annotation: = str

   The name of the C++ compiler found on the system (e.g., ``"g++"``, ``"cl.exe"``, or ``"clang"``). Returns ``None`` if no compiler was found.

Utilities
---------

.. autofunction:: clear_kernel_cache