.. _benchmarks:

Performance & Benchmarks
========================

To demonstrate the potential of the project, HeteroSymNN was tested against the Google JAX and Meta PyTorch frameworks across 4 distinct scenarios:

1. Increasing Heterogeneity in a layer.
2. VRAM/RAM scaling with batch sizes.
3. General Computing (Wide & Shallow vs. Deep & Narrow Homogeneous Networks).
4. Function constant mutation.

Additionally, a fifth internal benchmark is included to demonstrate the framework's hardware agnosticism and backend performance.

Testing Environment
--------------------
The tests were conducted on Windows WSL2 environment (Ubuntu), utilizing an Intel Core i9-13900HX CPU and an NVIDIA RTX 4060 Laptop GPU (8GB VRAM) running CUDA 13.x.

Each scenario was executed three times in isolation. We report the mean and standard deviation to eliminate potential OS background noise and thermal throttling artifacts.

**Methodological Rigor (Fair-Play Benchmarking)**

To ensure objective and defensible data, the underlying test script enforced strict operational constraints:

* **JAX VRAM Allocation Restrictions:** JAX's default behavior of aggressively preallocating 90% of available GPU VRAM was explicitly disabled (`XLA_PYTHON_CLIENT_PREALLOCATE="false"`). This ensures fair and accurate memory footprint comparisons, which is especially crucial for the Batch Size benchmark.
* **Asynchronous Execution Synchronization:** Hardware-level barriers (`torch.cuda.synchronize()`, `jax.block_until_ready()`, and `cp.cuda.get_current_stream().synchronize()`) were enforced before stopping timers. This guarantees we measured true sustained hardware execution latency, not merely Python's asynchronous dispatch time.
* **Warmup Passes:** Unmeasured forward and backward passes were executed prior to starting the timers. This isolates the frameworks' cold-start, lazy initialization, and initial memory allocation penalties from the actual computation benchmark.
* **Strict Memory Isolation:** Aggressive garbage collection and VRAM purging (`torch.cuda.empty_cache()`, `jax.clear_caches()`, etc.) were forced between framework passes to prevent memory fragmentation and cross-contamination.
* **JAX Optimization (Anti-Sandbagging):** To give the JAX baseline its optimal "best-case" performance, its execution loops were heavily optimized using the static compilation (`@partial(jax.jit)` and `jax.lax.scan`), ensuring maximum cache hits and completely unrolled C++ execution.
* **Kernel Compilation Caching:** While active VRAM and object states are purged between runs, compiled binary caches (e.g., CuPy's NVRTC temp-disk cache, JAX's XLA RAM cache) persist across the script's sequential execution. Consequently, similar graph structures or mathematical operations evaluated in later scenarios benefit from compilation cache hits. This causes *compile times* to progressively skew lower as the benchmark advances, which accurately reflects a real-world, iterative development lifecycle where frameworks reuse previously built kernels.
* **Sequential Hardware Caching:** Configurations and trials are executed sequentially. While software-level memory pools are actively purged, the underlying GPU instruction caches (L1/L2/L3) and the CUDA context remain "warm." This sequential design intentionally allows all frameworks to benefit from low-level hardware cache hits, ensuring the reported execution metrics reflect optimal, steady-state throughput rather than anomalous cold-boot latencies.

Benchmark Scenarios
-------------------

1. Degree of Heterogeneity
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Setup:** This benchmark utilizes a single hidden layer (1024 neurons) partitioned into `N` distinct mathematical functions (Total network parameters: 264,193). 

.. list-table:: VRAM and Execution Time vs. Unique Functions
   :header-rows: 1

   * - Unique Functions
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

**Analysis:** Both PyTorch and JAX exhibit scaling execution times as graph fragmentation increases, struggling with internal tensor slicing and concatenation. In contrast, HeteroSymNN demonstrates an `O(1)` scaling of VRAM usage relative to layer slicing, a direct result of its single-kernel function switching architecture.


2. Batch Size Scaling
^^^^^^^^^^^^^^^^^^^^^

**Setup:** This scenario evaluates how VRAM usage scales with batch size in a 4-function single-hidden-layer network (Total network parameters: 66,561). 

.. list-table:: VRAM Reduction vs. PyTorch
    :header-rows: 1

    * - Batch Size
      - PyTorch
      - H-SymNN
      - VRAM Reduction 
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

**Analysis:** Because HeteroSymNN relies on a single fused kernel implementation, there is no requirement to save intermediate routing results. This drops VRAM usage by ~76% at smaller batch sizes and ~44% at massive batch sizes. Consequently, devices with limited VRAM—such as laptops and edge devices (e.g., Raspberry Pi)—can confidently run batch sizes approximately twice as large as PyTorch.


3. General Computing
^^^^^^^^^^^^^^^^^^^^

**Setup:** Here we test general computation overhead for homogeneous networks across wide and deep architectures. 

.. list-table:: Wide vs. Deep Architecture Performance
    :header-rows: 1

    * - Architecture
      - Params
      - PyTorch (ms)
      - JAX XLA (ms)
      - H-SymNN Compile
      - H-SymNN (ms)
    * - Wide & Shallow (1x1024)
      - 264,193
      - 1.89 ± 0.45 ms
      - 1.33 ± 0.17 ms
      - 0.012 s
      - 3.01 ± 2.31 ms
    * - Deep & Narrow (16x64)
      - 78,913
      - 3.18 ± 0.32 ms
      - 0.83 ± 0.01 ms
      - 0.019 s
      - 5.14 ± 0.09 ms

**Analysis:** On standard, static architectures compiled entirely into C++ (like `jax.lax.scan`), Google JAX is heavily optimized. HeteroSymNN trades a few milliseconds here for Python loop dispatch overhead. However, this is an artifact of the current implementation, not an architectural ceiling. By logically decoupling the execution routing loop from the JIT-compiled kernels, HeteroSymNN intentionally avoids fusing the entire training process into a rigid static graph. Porting this decoupled routing logic to a native C++ object in the future will minimize this dispatch overhead while retaining the immense flexibility demonstrated in the other benchmarks.


4. Mutation Frequency
^^^^^^^^^^^^^^^^^^^^^

**Setup:** This test compares compilation and execution latency when a constant within an activation function is dynamically mutated. We used `sin(x * alpha)`, where `alpha` acts as the mutating constant. 

.. list-table:: Impact of Dynamic Graph Topology Changes
    :header-rows: 1

    * - Mutation Freq
      - Google JAX (ms/batch)
      - HeteroSymNN (ms/batch)
    * - Static
      - 1.63 ± 0.12 ms
      - 1.81 ± 0.55 ms
    * - Medium (1/10 batches)
      - 20.51 ± 26.08 ms
      - 1.69 ± 0.06 ms
    * - Extreme (Every batch)
      - 168.16 ± 234.57 ms
      - 4.97 ± 0.11 ms

**Analysis:** Static compilers like Google's XLA (JAX) cannot efficiently survive mutating architectures. When modifying graph topology dynamically, JAX treats structural changes as cache misses, triggering catastrophic C++ recompilation lockups (evidenced by the massive ± 234.57 ms stutter). Conversely, HeteroSymNN injects dynamic constants directly into active memory slots independently of the compiled kernel. This allows it to execute Liquid/Evolutionary networks with high consistency (± 0.11 ms) and over 33x faster than JAX under extreme mutation conditions.


5. Hardware Agnosticism (Internal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Setup:** To guarantee execution on any hardware, HeteroSymNN features three computational backends that the framework selects automatically based on system capabilities. 

.. list-table:: Backend Performance Comparison
    :header-rows: 1

    * - Execution Mode
      - Backend Tech
      - Compile Time
      - Execution (ms/batch)
    * - GPU_CUDA
      - NVRTC (In-Memory)
      - 0.0024 s
      - 1.64 ± 0.16 ms
    * - CPU_JIT
      - GCC/MSVC Subprocess
      - 0.7857 s
      - 68.15 ± 28.77 ms 
    * - CPU_PYTHON
      - Pure NumPy 
      - 0.0196 s
      - 2.70 ± 0.35 ms

**Analysis:** As observed above, the `CPU_JIT` backend currently underperforms compared to the other two methods. Therefore, for the 0.3.0 version release, `CPU_JIT` will be deactivated, making `CPU_PYTHON` the default fallback for all CPU computations. By leveraging the NumPy architecture, the performance drop from GPU to CPU remains highly manageable (less than a 2x latency increase), ensuring robust accessibility and deployment safety across all devices.