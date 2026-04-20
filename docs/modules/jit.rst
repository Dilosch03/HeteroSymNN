.. _jit:

JIT Compiler & Code Generation
==============================

The :class:`~HeteroSymNN.JIT.compiler.SymbolicJITCompiler` is the computational heart of HeteroSymNN. 

Rather than relying on pre-compiled, static mathematical functions (like a hardcoded PyTorch ReLU), HeteroSymNN operates as a **Differentiable Compiler**. It accepts high-level symbolic definitions as raw strings, symbolically computes their exact mathematical derivatives, and compiles them into highly optimized, hardware-specific executable kernels at runtime.

The Compilation Pipeline
------------------------
When a network initializes or a custom loss function is defined, the JIT compiler executes a strict 4-step pipeline:

1. **Parse & Derive:** The compiler reads the user's mathematical string (e.g., ``"sin(num * alpha)"``) and converts it into an Abstract Syntax Tree (AST) using SymPy, simultaneously calculating the exact symbolic derivative for the backward pass.
   
   .. warning::
      **Security Note (The AST Sandbox):** The explicit AST whitelist sandbox—designed to strictly prevent Arbitrary Code Execution (ACE) via malicious SymPy object injection—is currently in active development for a future release. While SymPy parsing is inherently safer than raw ``eval()``, you should not load ``.symnn`` files from untrusted sources in the current version.

2. **Code Generation:** The AST is passed through the internal ``codegen`` module, which acts as the translation dictionary for the hardware.
3. **Kernel Fusion:** To prevent the massive performance penalty of "kernel launch overhead," the compiler fuses all of the distinct mathematical instructions for every neuron in a layer into a single, unified C++ ``switch`` statement.
4. **Execution & Caching:** The raw C++ string is compiled using the active hardware backend. The resulting binary execution pointer is securely cached to disk to guarantee instantaneous loading on future runs.

The Code Generation Addon (codegen)
-----------------------------------
Operating as an essential subsystem within the general JIT flow, the internal ``codegen`` module acts as the framework's blueprint library. Once the compiler has processed the AST, it relies on this module for the final translation:

* **Function Converters:** It contains the direct dictionaries that map standard SymPy operations into their hardware-specific syntax (e.g., translating a SymPy ``Max`` into a CUDA ``fmaxf`` or C++ ``std::max``).
* **Execution Templates:** It holds the raw C++ and CUDA boilerplate string templates required to scaffold the final fused ``forward`` and ``backward`` passes.
* **Extensibility:** Currently, it houses a robust collection of the most commonly used mathematical functions. In future updates, these translation dictionaries will be exposed directly to the user API, allowing researchers to inject completely custom C++/CUDA hardware instructions and map them to their own SymPy shortcut symbols.

Hardware Agnosticism (The Fallback Engine)
------------------------------------------
The compiler guarantees execution regardless of your hardware environment by automatically routing the generated C++ code to one of three backends:

* **``GPU_CUDA`` (Maximum Performance):** Pushes the generated string to CuPy, leveraging NVIDIA's NVRTC to compile the custom math into a massively parallel GPU kernel instantly.
* **``CPU_JIT`` (High Performance):** Detects local C++ compilers (GCC/MSVC) and uses OpenMP to build native, multi-threaded CPU binaries.
* **``CPU_PYTHON`` (The Safety Net):** If no C++ compilers or GPUs are found, the engine gracefully falls back to generating optimized Python Lambdas and NumPy matrix masking, ensuring the code remains functional on any machine.

Zero-Recompile Tuning (Dynamic Constants)
-----------------------------------------
A major limitation of standard JIT compilers (like JAX's XLA) is that altering architectural constants forces a complete graph recompilation. 

HeteroSymNN treats all symbolic constants (like ``alpha`` or ``beta``) as **mutable kernel arguments**. You can update these hyperparameters dynamically on the fly (e.g., during a Grid Search or Evolutionary mutation) and the JIT compiler will simply update the memory pointer, bypassing the compilation phase entirely and executing 5x to 10x faster than traditional tracing compilers.

.. note::
   Currently the reassignment of values of the dynamic constants is unoptimized and has a greater impact in performance, due to device transfer speeds and overhead and sequential assignment. 

API Reference
-------------

.. currentmodule:: HeteroSymNN.JIT.compiler

.. autoclass:: SymbolicJITCompiler
   :members: 
   :undoc-members:
   :private-members: _compile_cpp_kernels, _compile_py_kernels, _compile_cuda_kernels, _change_method