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

