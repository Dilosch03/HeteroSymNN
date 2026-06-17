.. _roadmap:

Development Roadmap
===================

HeteroSymNN is currently in active beta. The architecture is evolving towards a secure, modular, and extensible ecosystem.

Current Phase: v0.3.x (API Stabilization)
-----------------------------------------
* **[Completed]** Restructure of the project for scalability and modularity.

Upcoming: v0.4.0 (API, Persistence & Heterogeneous Systems)
-----------------------------------------------------------
* **API Quality of Life** (SSOT Enforcement):

  * Converting magic routing strings (like ``"GPU_CUDA"``) into strict String-Enums to enable IDE autocomplete and prevent typos.

* **JIT Compiler Evolution**: Implementing string safeguards.

  * Making a modular architecture for expansion of capabilities.

* **Custom class Persistence:** Adding a hierarchical registry loading of custom classes.
* **Expansion of classes:** Adding more types of initializers, optimizers, losses and data transformers.
* **Expanding the CLI:** Adding more commands to the CLI to make it more useful like a ``build`` command for generating .symnn files form a json defined structure.
* **Global Registry:** Implementing the Global Function Registry (``settings.register_functions()``) for alias expansion before SymPy compilation.
* **Heterogeneous Optimizers**: Adding the ability to set per-neuron the type of optimizer that is going to be used to update it.
* **Heterogeneous Initializer**: Adding the ability to set different initializer functions depending on the data type being requested.
* **Multi objective loss functions:** Adding the ability to construct one loss class that contains multiple loss functions for each output of the network.
* **RAM Utilization Calculator:** Adding a memory estimation calculation before execution for OOM (Out Of Memory) protection.
* **Generalization of function definitions:** Moving constant extraction and string expansion into the :customref:`SymbolicJITCompiler <jit>` so any class can accept :type:`~HeteroSymNN.types.FlexibleNodeConfig`.

Upcoming: v0.5.0 (JIT Core, Scope & Data Pipeline)
--------------------------------------------------
* **Lexical Analyzer**: Analitical preprosses of the function strings for domain understanding of the expression.

* **Hierarchical Scope Resolution (4-Tier Constants):** Implementing a strict 4-tier scope resolution pipeline for the Lexical Analyzer to gracefully handle variables during Symbolic JIT compilation and reduction of possible ram/vram usage.
 
  * **Tier 1 - Static (Baked-In):** Hard coded constants that will never change.
  * **Tier 2 - Global (Network-Wide):** Universal hyperparameters shared across the entire network.
  * **Tier 3 - Environment (Per-Layer):** Layer hyperparameter constant that will be shared for all neurons in the layer.
  * **Tier 4 - Per-Node (Current):** Current Implementation of the Dynamic constants.

* **Batch dependant constants:** Adding the ability to construct a constant that differed depending on the information or state in the sample and be batch compatibles.
* **Multi variable loss functions:** Adding the ability to construct a loss function that accepts multiple inputs (e.g., f(x,y,z,w)).
* **Asynchronous Data Bus**: Building the ``DatasetStreamer`` to feed the GPU in chunks without overflowing CPU RAM.
* **Modular Evaluation**: Moving the metrics calculations out of the wrappers and making them into their own modular methods.
* **Evaluation Suite:** Classes to get tendencies metrics of how an architecture is performing across multiple instances.
* **Constants Tuner:** Method or function to find the best values of the dynamic constant in the network.
  
Upcoming: v0.6.0 (Backend Optimization)
----------------------------------------
* **CPU_JIT Rewrite:** Make the computational method `CPU_JIT` better than native py by changing the compiler to be jit compatible and optimize the parallelization of the computation by focusing the architecture to use a SIMD flow.
* **Less loops:** Making the instancing and value updates less reliant in loops and more parallelization.
* **In-Place Memory Allocation:** Optimizing the forward and backward computation to eliminate temparary memory allocations.
* **String Parsing Optimization:** Optimizing the string parsing of the jit by eliminating redundant object creation.
* **GIL-less Compatibility:** Implementation of methods that optimize CPU bounded tasks when the python version has no GIL with automatic detection.
* **Memory managment for optimization:** Reorganization of the tensors structure to packet better similar functions and accelerate the computation.
  
Upcoming: v0.7.0 (Advanced Topologies & Hardware Control)
----------------------------------------------------------
* **Recurrent Networks (RNNs):** Implementing Backpropagation Through Time (BPTT) math and implement sequence base learning.
* **Internal Normalization:** Adding BatchNorm and LayerNorm natively into the JIT pipeline to prevent exploding gradients in deeper topologies.
* **Parameter Offloading:** Automated Disk/RAM/VRAM paging system to stream model parameters during computation, enabling the execution of massive models that exceed available memory.
* **Extrenal compatibility:** Adding functions and wrappers for layer compatibility with other frameworks. (posible optional external packet)
* **Learnable constants:** Adding the ability for the optimizers to tweak the value of the dynamic constants.
* **MoELayer:** Adding a new layer type that uses a Mixture of Experts architecture.

Upcoming: v0.8.0 (Dual-Engine Architecture)
--------------------------------------------
* **CUDA VM:** Building the Python Bytecode Assembler and a CUDA Virtual Machine to execute opcodes directly in VRAM without invoking the nvcc compiler.
* **Decided JIT:** Implementing the heuristic algorithm to decide if it's going to use the interpreter or the JIT-compiled kernels during training based on dynamic profiling.
* **High Performance Module:** Adding a module optilized for better gpu performance of the networks. (external optional packet)
* **N-Dimensions Support:** Upgrading the kernels to support N-Dimensional tensor operations.
* **Chunked Execution:** Slicing massive layer computations into sequential sub-blocks to prevent activation memory spikes during the forward/backward pass.

Future: v0.9.0+
----------------
* **Evolutionary Networks (EvoNets)**
  
  * *Buffer and Flush Architecture:* Mutating the entire next generation's math in CPU NumPy arrays, and flushing it to the GPU in a single massive PCIe payload.

* **Framework compatibility:** Adding functions and wrappers for high compatibility of the framework objects with other frameworks. (posible optional external packet)
* **Reinforcement Learning Suite:** Set up of the required states, methods, wrappers and tools for doing Reinforcement Learning. (posible optional external packet)
* Model Quantization.
* Compatibility with ONNX files.
* Full model binary compilation for edge devices. 
* AMD computation compatibility.
* Cluster computation.
* Network analyzers.
* Transformers and convolutional layers.