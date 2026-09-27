.. _benchmarks:

Performance & Benchmarks
========================

HeteroSymNN was compared against PyTorch and JAX in four scenarios, plus one internal comparison of its own backends:

1. Increasing heterogeneity (number of distinct activation functions) in one layer.
2. GPU memory as the batch size grows.
3. Standard homogeneous networks (wide & shallow vs. deep & narrow).
4. Changing an activation constant during training.
5. HeteroSymNN backends compared with each other.

**Summary.** HeteroSymNN's step time stays flat as a layer becomes more heterogeneous, and it uses less GPU memory than a straightforward PyTorch implementation of a mixed-activation layer. On standard homogeneous networks it is slower than both PyTorch and JAX. Choose it for flexibility, not raw speed.

Testing Environment
--------------------
The tests were run in WSL2 (Ubuntu) on an Intel Core i9-13900HX CPU and an NVIDIA RTX 4060 Laptop GPU (8 GB VRAM) with CUDA 13.x.

Each measurement is the mean ± standard deviation of repeated trials (100 training steps per trial, 5 trials unless stated otherwise). A training step is forward pass + loss + backward pass + parameter update.

Methodology
-----------

* **Synchronization:** ``torch.cuda.synchronize()``, ``jax.block_until_ready()`` and ``cp.cuda.get_current_stream().synchronize()`` were called before stopping each timer, so the times include the GPU work and not only Python's asynchronous dispatch.
* **Warmup:** Unmeasured steps were run before timing, so compilation, lazy initialization and first allocations are excluded from step times. HeteroSymNN compile times are reported separately where relevant.
* **Memory isolation:** Memory pools and caches were cleared between frameworks (``torch.cuda.empty_cache()``, ``jax.clear_caches()``, CuPy pool release). JAX's default preallocation of 90% of VRAM was disabled (``XLA_PYTHON_CLIENT_PREALLOCATE="false"``).
* **JAX:** Training loops were compiled with ``jax.jit``. Scenarios 1 and 3 also use ``jax.lax.scan``, so the whole loop runs as a single XLA program.
* **Compilation caches:** Compiled kernels (CuPy NVRTC disk cache, JAX XLA cache) persist across scenarios within one run, so compile times of later scenarios can be lower. Step times are not affected.

Known differences between the setups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read the results with these differences in mind:

* **Optimizer:** PyTorch and HeteroSymNN use Adam. The JAX baselines use a plain SGD update, which does less work per step.
* **Memory measurement:** PyTorch memory is ``torch.cuda.max_memory_allocated()`` (peak, includes the input batch). HeteroSymNN memory is the drop in free device memory (``cudaMemGetInfo``) around model creation and one training step, measured after the input batch was already on the GPU. The two numbers are indicative, not directly equivalent.
* **PyTorch baseline for heterogeneous layers:** Implemented the straightforward way: slice the pre-activations per function, apply each function, concatenate. Other implementations (masking, ``torch.compile``) could behave differently.
* **Scenario 4 JAX setup:** The constant was declared as a static (compile-time) argument. This forces a recompilation for every new value. See the scenario for details.

Benchmark Scenarios
-------------------

1. Degree of Heterogeneity
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Setup:** One hidden layer of 1,024 neurons (256 → 1024 → 1, 264,193 parameters, batch 2,048), split into `N` groups that each use a different activation function.

.. list-table:: Memory and step time vs. number of distinct functions
   :header-rows: 1

   * - Distinct Functions
     - PyTorch VRAM
     - PyTorch (ms)
     - JAX XLA (ms)
     - H-SymNN VRAM
     - H-SymNN (ms)
   * - **1**
     - 47.3 MB
     - 1.45 ± 0.94
     - 0.25 ± 0.00
     - 31.1 MB
     - 1.42 ± 0.10
   * - **2**
     - 53.3 MB
     - 0.97 ± 0.05
     - 0.29 ± 0.00
     - 31.1 MB
     - 1.44 ± 0.12
   * - **4**
     - 53.3 MB
     - 0.95 ± 0.02
     - 0.69 ± 0.00
     - 31.1 MB
     - 1.35 ± 0.02
   * - **8**
     - 53.3 MB
     - 1.22 ± 0.03
     - 0.63 ± 0.01
     - 31.1 MB
     - 1.59 ± 0.11
   * - **16**
     - 53.3 MB
     - 1.61 ± 0.03
     - 0.35 ± 0.00
     - 31.1 MB
     - 1.25 ± 0.12
   * - **32**
     - 53.3 MB
     - 2.37 ± 0.06
     - 1.52 ± 0.17
     - 31.1 MB
     - 1.28 ± 0.02

**Analysis:** HeteroSymNN's step time stays between 1.25 and 1.59 ms and its memory stays at 31.1 MB whatever the number of functions. All functions of the layer live in one kernel, so more functions do not mean more kernels or intermediate tensors. PyTorch slows down as the layer is split into more slices (0.95 → 2.37 ms), and JAX varies with the fragmentation but is faster than HeteroSymNN up to 16 functions.

In practice: with fewer than ~16 distinct functions per layer, PyTorch and JAX are faster. From ~16 (PyTorch) or ~32 (JAX) distinct functions, HeteroSymNN is the fastest of the three here.


2. Batch Size Scaling
^^^^^^^^^^^^^^^^^^^^^

**Setup:** GPU memory for a 4-function single-hidden-layer network (128 → 512 → 1, 66,561 parameters) as the batch size grows. PyTorch was measured over one forward/backward pass. HeteroSymNN was measured over model creation plus one full training step with Adam (see *Known differences*).

.. list-table:: GPU memory vs. batch size
    :header-rows: 1

    * - Batch Size
      - PyTorch
      - H-SymNN
      - Difference
    * - 1,024
      - 33.1 MB
      - 7.8 MB
      - -76%
    * - 4,096
      - 67.6 MB
      - 27.3 MB
      - -60%
    * - 16,384
      - 205.7 MB
      - 105.5 MB
      - -49%
    * - 65,536
      - 758.0 MB
      - 418.0 MB
      - -45%
    * - 131,072
      - 1494.5 MB
      - 834.8 MB
      - -44%

**Analysis:** The slice-and-concatenate PyTorch layer keeps extra copies of the pre-activations (one per slice plus the concatenated result) for the backward pass. HeteroSymNN keeps a fixed set of buffers per layer (pre-activation, activation and delta), whatever the number of functions. The gap is largest at small batch sizes (-76%), where fixed overheads dominate, and settles around -44% at large batch sizes. With the same memory budget, this allows roughly 1.8× larger batches for this kind of layer.


3. General Computing
^^^^^^^^^^^^^^^^^^^^

**Setup:** Homogeneous networks (one activation function everywhere), batch 2,048.

.. list-table:: Wide vs. deep homogeneous networks
    :header-rows: 1

    * - Architecture
      - Params
      - PyTorch (ms)
      - JAX XLA (ms)
      - H-SymNN Compile
      - H-SymNN (ms)
    * - Wide & Shallow (1x1024)
      - 264,193
      - 1.89 ± 0.45
      - 1.33 ± 0.17
      - 0.012 s
      - 3.01 ± 2.31
    * - Deep & Narrow (16x64)
      - 78,913
      - 3.18 ± 0.32
      - 0.83 ± 0.01
      - 0.019 s
      - 5.14 ± 0.09

**Analysis:** On standard networks HeteroSymNN is about 1.6× slower than PyTorch and 2–6× slower than JAX. The gap is largest on the deep network, where a step runs many small layers and the per-layer Python dispatch dominates. JAX compiles the entire training loop into one program (``jax.lax.scan``), which removes that overhead completely. Moving the layer loop out of Python is on the roadmap. Until then, for homogeneous networks PyTorch or JAX are the better choice.


4. Changing a Constant During Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Setup:** A 128 → 256 → 1 network with the activation ``sin(x * alpha)``, batch 1,024. `alpha` is changed on all 256 neurons never, every 10th batch, or every batch.

.. list-table:: Step time vs. how often the constant changes
    :header-rows: 1

    * - Change Frequency
      - JAX, alpha as static argument (ms/step)
      - HeteroSymNN (ms/step)
    * - Never
      - 1.63 ± 0.12
      - 1.81 ± 0.55
    * - Every 10 batches
      - 20.51 ± 26.08
      - 1.69 ± 0.06
    * - Every batch
      - 168.16 ± 234.57
      - 4.97 ± 0.11

**Analysis:** In HeteroSymNN, constants are kernel arguments, so changing them never triggers a recompilation. The extra cost when changing all 256 constants every batch (~3 ms) is the ``change_constants`` call itself: it writes the constants one at a time, which on the GPU means one small host-to-device copy per constant.

The JAX column shows what happens when a value is baked into the compiled program: every new value is a cache miss and a full XLA recompilation, hence the large and unstable times. This is **not** the idiomatic JAX setup. Passing `alpha` as a normal (traced) argument compiles once and reuses the program for every value. A CPU check of the same step confirms this: 30 changes of a static `alpha` caused 30 compilations, while a traced `alpha` compiled once. PyTorch's eager mode doesn't compile at all, so it has no recompilation cost either.

The fair takeaway: HeteroSymNN makes *every* constant in a formula a runtime value automatically, per neuron, without the user having to design for it. Changing constants does not force recompilation in PyTorch eager or in idiomatic JAX either. The GPU numbers for traced JAX and for PyTorch will be added to this table in the next benchmark run.


5. Backend Comparison (Internal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Setup:** The same small network (64 → 128 → 1, batch 1,024) on each HeteroSymNN backend.

.. list-table:: Backend Performance Comparison
    :header-rows: 1

    * - Execution Mode
      - Backend Tech
      - Compile Time
      - Execution (ms/step)
    * - GPU_CUDA
      - NVRTC (In-Memory)
      - 0.0024 s
      - 1.64 ± 0.16
    * - CPU_JIT
      - GCC/MSVC Subprocess
      - 0.7857 s
      - 68.15 ± 28.77
    * - CPU_PYTHON
      - Pure NumPy
      - 0.0196 s
      - 2.70 ± 0.35

**Analysis:** The C++/OpenMP backend (``CPU_JIT``) was far slower than plain NumPy and slow to compile, so it is disabled from v0.3.0. Requests for it fall back to ``CPU_PYTHON``.

On this small network, NumPy is within 2× of the GPU, because there is little work per step for the GPU to parallelize. The gap grows with network and batch size. Also note that on ``CPU_PYTHON``, layers with several distinct functions currently evaluate each neuron separately, so mixed layers are relatively slower on CPU than homogeneous ones.
