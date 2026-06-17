.. _jit:

JIT Compiler & Code Generation
==============================

The :class:`~HeteroSymNN.JIT.compiler.SymbolicJITCompiler` is the computational heart of HeteroSymNN. 

Rather than relying on pre-compiled, static mathematical functions (like a hardcoded PyTorch ReLU), HeteroSymNN operates as a **Differentiable Compiler**. It accepts high-level symbolic definitions as raw strings, symbolically computes their exact mathematical derivatives, and compiles them into optimized, hardware-specific executable kernels at runtime.

The Compilation Pipeline
------------------------
When a network initializes or a custom loss/activation function is defined, the JIT compiler executes a strict 4-step pipeline:

1. **Parse & Derive:** The compiler parses the user's mathematical string (e.g., ``"sin(num * alpha)"``) into an Abstract Syntax Tree (AST). It extracts free variables as dynamic constants and provides automatic alias resolution (recognizing ``num``, ``x``, or ``z`` as the main activation variable). Then, it calculates the exact symbolic derivative using SymPy for the backward pass.
2. **Code Generation:** The parsed SymPy expression is passed through the internal ``codegen`` module, which acts as the translation dictionary for the hardware, mapping symbolic math to C++/CUDA operations.
3. **Kernel Fusion:** To prevent the massive performance penalty of "kernel launch overhead," the compiler fuses all of the distinct mathematical instructions for every neuron in a layer into a single, unified C++ ``switch`` statement.
4. **Execution & Caching:** The raw C++ string is compiled using the active hardware backend. The resulting binary execution pointer is securely cached to disk (using SHA-256 hashes of the configuration) to guarantee instantaneous loading on future runs. Memory-based kernel caches are also used to prevent redundant parsing.

.. warning::
   **Security Note:** Although the parser is robust and strictly evaluates expressions via an Abstract Syntax Tree (AST), the underlying symbolic engine is complex and there is still a possibility of arbitrary code injection. You should be careful and **never load external ``.symnn`` files from untrusted sources**.

The Code Generation Addon (codegen)
-----------------------------------
Operating as an essential subsystem within the general JIT flow, the internal ``codegen`` module acts as the framework's blueprint library. Once the compiler has parsed the equation, it relies on this module for the final translation:

* **Function Converters:** It contains the direct dictionaries that map standard SymPy operations into their hardware-specific syntax (e.g., translating a SymPy ``Max`` into a CUDA ``fmaxf`` or C++ ``std::max``).
* **Common Formulas:** It holds optimized pre-defined formulas for common activations (e.g., ``relu``, ``mish``, ``gelu``) and losses (e.g., ``mse``, ``bce``), allowing users to reference them by their standard names.
* **Execution Templates:** It holds the raw C++ and CUDA boilerplate string templates required to scaffold the final fused ``forward`` and ``backward`` passes.
* **Extensibility:** In future updates, these translation dictionaries will be exposed to users, allowing researchers to expand with compatible C++/CUDA hardware instructions and map them to their own function aliases.

Hardware Agnosticism (The Fallback Engine)
------------------------------------------
The compiler guarantees execution regardless of your hardware environment by automatically routing the generated C++ code to one of three backends:

* **``GPU_CUDA`` (Maximum Performance):** Pushes the generated string to CuPy, leveraging NVIDIA's NVRTC to compile the custom math into a massively parallel GPU kernel instantly.
* **``CPU_JIT`` (High Performance):** Detects local C++ compilers (GCC/MSVC) and uses OpenMP to build native, multi-threaded CPU binaries as shared libraries.
* **``CPU_PYTHON`` (The Safety Net):** If no C++ compilers or GPUs are found, the engine gracefully falls back to generating optimized Python Lambdas using NumPy, ensuring the code remains functional on any machine.

Zero-Recompile Tuning (Dynamic Constants)
-----------------------------------------
A major limitation of standard JIT compilers (like JAX's XLA) is that altering architectural constants forces a complete graph recompilation. 

HeteroSymNN treats all symbolic constants (like ``alpha`` or ``beta``) as **mutable kernel arguments**. Through an optimized contiguous parameter array and offset pointers, you can update these hyperparameters dynamically on the fly (e.g., during a Grid Search or Evolutionary mutation). The JIT compiler simply reads the updated values from memory during execution, bypassing the compilation phase entirely and executing significantly faster than traditional tracing compilers.

API Reference
-------------

.. currentmodule:: HeteroSymNN.JIT.compiler

.. autoclass:: SymbolicJITCompiler
   :members: 
   :undoc-members:
   :private-members: _get_ccode_from_config, _generate_kernel_artifacts, _compile_cpp_kernels, _compile_py_kernels, _compile_cuda_kernels, _change_method