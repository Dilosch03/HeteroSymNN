.. _jit:

JIT Compiler
============

The ``SymbolicJITCompiler`` is the computational heart of HeteroSymNN. It is responsible for transforming high-level symbolic definitions of mathematical functions (activations and losses) into highly optimized, hardware-specific executable kernels at runtime.

This compiler bridges the gap between flexibility and performance by leveraging SymPy for symbolic differentiation and code generation, and then compiling that code into:

* **CUDA Kernels (GPU_CUDA):** For massive parallelism on NVIDIA GPUs using CuPy.
* **C++ Shared Libraries (CPU_JIT):** For high-performance CPU execution using OpenMP and system compilers (MSVC/GCC).
* **Python Lambdas (CPU_PYTHON):** As a fallback for maximum compatibility.

It handles the automatic differentiation of user-defined formulas, manages the compilation cache to avoid redundant work, and provides a unified interface (``forward_kernel``, ``backward_kernel``) for the rest of the framework.

.. currentmodule:: HeteroSymNN.JIT.compiler

.. autoclass:: SymbolicJITCompiler
   :members: set_gpu_id, _change_method
   :private-members: _compile_cpp_kernels, _compile_py_kernels, _compile_cuda_kernels
   :undoc-members: