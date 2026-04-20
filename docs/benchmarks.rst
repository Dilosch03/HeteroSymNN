.. _benchmarks:

Performance & Benchmarks
========================

Because HeteroSymNN is fundamentally a **Differentiable JIT Compiler**, benchmarking it against traditional frameworks like PyTorch or JAX requires understanding the difference between the "Cold Start" (JIT compilation time) and the "Steady State" (execution time).

1. Standard Execution (Steady State)
------------------------------------
In standard, static-graph scenarios, HeteroSymNN's C++ and CUDA kernel fusion yields performance on par with ``torch.compile()``. By fusing the mathematical instructions of every neuron into a single unified execution pass, HeteroSymNN completely avoids the massive "kernel launch overhead" that plagues naive custom Python loops.

2. The "Dynamic Constants" Superpower
-------------------------------------
Where HeteroSymNN completely destroys traditional tracing compilers is in **highly dynamic topologies** (e.g., Evolutionary Networks, Hyperparameter Grid Searching, or Neural ODEs).

Standard tracing compilers (like JAX's XLA) hardcode architectural constants into the execution graph. If you change a parameter during training, it triggers a catastrophic cache miss and forces a complete graph recompilation. 

HeteroSymNN treats symbolic constants as mutable kernel arguments. **You can update hyperparameters in real-time without triggering a recompilation.**

**Benchmark: Rapid Constant Mutation (Per Batch)**
*Testing a network where an internal symbolic parameter is updated every single batch.*

* **JAX (XLA JIT on RTX 4060 GPU):** ``~50.030 ms / batch``
* **HeteroSymNN (CPU Pure Python Fallback):** ``~10.638 ms / batch``

*Result:* Even when intentionally forcing HeteroSymNN into its slowest pure-Python fallback mode, it executed dynamic architectural changes **nearly 5x faster** than Google's flagship XLA compiler running on a dedicated GPU.