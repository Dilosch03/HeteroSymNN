.. _hardware-backend:

Hardware & Backends
===================

The Concept
-----------
The ``hardware`` module is the first code executed when you import HeteroSymNN. Before any math is parsed or networks are built, this module dynamically probes your local operating system.

It searches for CUDA-compatible NVIDIA GPUs, verifies CuPy installations, and scans the system PATH for valid C++ compilers (like GCC, Clang, or MSVC). Based on what it finds, it sets global boolean flags that the rest of the framework uses to safely route memory, initialize arrays, and determine the optimal execution backend.

When to Use (Debugging Hardware)
--------------------------------
Because hardware routing is fully automated by the global :customref:`settings <configuration>` object, you usually do not need to interact with this module directly. 

However, if your network is running significantly slower than expected (e.g., it is silently falling back to the pure Python safety net), you should use this module to **debug your hardware detection**. By inspecting these flags, you can quickly determine if the framework failed to detect your GPU or C++ compiler due to missing PATH variables or broken package dependencies.

Code Example
------------
Here is a simple diagnostic script you can run to verify that HeteroSymNN has correctly identified your hardware capabilities before starting a massive training run:

.. code-block:: python

    from HeteroSymNN.Backend import hardware

    print("--- HeteroSymNN Hardware Diagnostics ---")
    print(f"GPU / CUDA Detected: {hardware.GPU_ENABLED}")
    print(f"Number of GPUs:      {hardware.NUM_GPUS}")
    print(f"C++ JIT Compiler:    {hardware.CPP_JIT_ENABLED}")
    print(f"CPU Threads Avail:   {hardware.NUM_CPU_THREADS}")


API Reference
-------------

.. currentmodule:: HeteroSymNN.Backend.hardware

.. autodata:: GPU_ENABLED
   :annotation: = bool

   ``True`` if a CUDA-capable GPU is detected and CuPy is correctly installed in the Python environment.

.. autodata:: CPP_JIT_ENABLED
   :annotation: = bool

   ``True`` if a compatible C++ compiler (g++, clang, or cl.exe) is successfully found in the system PATH and responds to version checks.

.. autodata:: NUM_GPUS
   :annotation: = int

   The total number of available CUDA devices detected by the CuPy runtime.

.. autodata:: NUM_CPU_THREADS
   :annotation: = int
   
   The maximum number of logical CPU threads available on the host machine.