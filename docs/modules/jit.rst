JIT Compiler
============

The **JIT** (Just-In-Time) module is the engine of HeteroSymNN. It handles the translation of symbolic math expressions into optimized machine code.

.. automodule:: HeteroSymNN.JIT.compiler
   :members:
   :undoc-members:
   :show-inheritance:

Symbolic Compiler
-----------------

.. autoclass:: HeteroSymNN.JIT.compiler.SymbolicJITCompiler
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

   **Capabilities:**

   * **Parses** SymPy expressions from string configs.
   * **Differentiates** expressions symbolically for backpropagation.
   * **Compiles** to C++ (OpenMP) or CUDA kernels depending on the hardware.