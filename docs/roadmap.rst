.. _roadmap:

Development Roadmap
===================

HeteroSymNN is currently in active beta. The architecture is evolving towards a highly secure, modular, and extensible ecosystem.

Current Phase: v0.3.x (API Stabilization)
-----------------------------------------
* **[Completed]** Restructure of the project for scalability and modularity.

Upcoming: v0.4.0 (Security & Persistence)
-----------------------------------------
* **Lexical Analyzer & Global Registry**: Moving constant extraction and string expansion into the ``SymbolicJITCompiler``.

  * Implementing the Global Function Registry (``settings.register_funtions()``) for alias expansion before SymPy compilation.

* **API Quality of Life** (SSOT Enforcement):

  * Converting magic routing strings (like ``"GPU_CUDA"``) into strict String-Enums to enable IDE autocomplete and prevent typos.

* **Asynchronous Data Bus**: Building the ``DatasetStreamer`` to feed the GPU in chunks without overflowing CPU RAM.
* **JIT Compiler Evolution**: Implementing string safeguards.

  * Making a modular architecture for expansion of capabilities.

* **Modular Evaluation**: Moving the metrics calculations out of the wrappers and making them into their own modular methods.
* **Internal Normalization**: Adding internal layer normalization methods for numerical stability.
* **Memory Orchestrators**: Advanced automated RAM/VRAM paging and methods for clearing temporary memory.
* **Heterogeneous Optimizers**: Adding the ability to set per-neuron the type of optimizer that is going to be used to update it.
* **Heterogeneous Initializer**: Adding the ability to set different initializer functions depending on the data type being requested.
* **Hierarchical Scope Resolution (4-Tier Constants):** Implementing a strict 4-tier scope resolution pipeline for the Lexical Analyzer to gracefully handle variables during Symbolic JIT compilation and reduction of possible ram/vram usage.
 
  * **Tier 1 - Static (Baked-In):** Hard coded constants that will never change.
  * **Tier 2 - Global (Network-Wide):** Universal hyperparameters shared across the entire network.
  * **Tier 3 - Environment (Per-Layer):** Layer hyperparameter constant that will be shared for all neurons in the layer.
  * **Tier 4 - Per-Node (Current):** Current Implementation of the Dynamic constants.
  
* **Evaluation Suite:** Classes to get tendencies metrics of how an architecture is performing across multiple instances.
* **Constants Tuner:** Method or function to find the best values of the dynamic constant in the network.
* **Expansion of classes:** Adding more types of initializers, optimizers, losses and data transformers.
* **Custom class Persistence:** Adding a hierarchical registry loading of custom classes.
* **Expanding the CLI:** Adding more commands to the CLI to make it more useful like a ``build`` command for generating .symnn files form a json defined structure.
  
Upcoming: v0.5.0 (Backend Optimization)
----------------------------------------
* **CPU_JIT Rewrite:** Make the computational method `CPU_JIT` better than native py by changing the compiler to be jit compatible and optimize the parallelization of the computation by focusing the architecture to use a SIMD flow.
* **Less loops:** Making the instancing and value updates less reliant in loops and more parallelization.
* **In-Place Memory Allocation:** Optimizing the forward and backward computation to eliminate temparary memory allocations.
* **String Parsing Optimization:** Optimizing the string parsing of the jit by eliminating redundant object creation.
* **GIL-less Compatibility:** Implementation of methods that optimize CPU bounded tasks when the python version has no GIL with automatic detection.
* **Memory managment for optimization:** Reorganization of the tensors structure to packet better similar functions and accelerate the computation.
  
Upcoming: v0.6.0 (Advanced Topologies & Hardware Control)
----------------------------------------------------------
* **Recurrent Networks (RNNs):** Implementing Backpropagation Through Time (BPTT) math natively on the Heavy Forge (C++ JIT) to verify gradients perfectly.
* **Internal Normalization:** Adding BatchNorm and LayerNorm natively into the JIT pipeline to prevent exploding gradients in deeper topologies.
* **The Memory Orchestrator:** Advanced automated RAM/VRAM paging.
* **Extrenal compatibility:** Adding functions and wrappers for compatibility of the networks with other frameworks.
* **Learnable constants:** Adding the ability for the optimizers to tweak the value of the dynamic constants.
* **MoELayer:** Adding a new layer type that uses a Mixture of Experts architecture.

Upcoming: v0.7.0 (Dual-Engine Architecture)
--------------------------------------------
* **CUDA VM:** Building the Python Bytecode Assembler and a CUDA Virtual Machine to execute opcodes directly in VRAM without invoking the nvcc compiler.
* **Decided JIT:** Implementing the heuristic algorithm to decide if it's going to use the interpreter or the JIT-compiled kernels during training based on dynamic profiling.
* **High Performance Module:** Adding a module optilized for better gpu performance of the networks.

Future: v0.8.0+
----------------
* **Evolutionary Networks (EvoNets)**
  
  * *Buffer and Flush Architecture:* Implementing the ``_is_dirty`` flag. Mutating the entire next generation's math in CPU NumPy arrays, and flushing it to the GPU in a single massive PCIe payload.

* **N-Dimensional Support:** Upgrading the kernels to support the Z-axis (Population Index) and executing evaluations via Batched Matrix Multiplication (BMM).
* **Framework compatibility:** Adding functions and wrappers for compatibility of the networks with other frameworks.
* **Reinforcement Learning Suite:** Set up of the required states, methods, wrappers and tooks for doing Reinforcement Learning.
* Model Quantization.
* Compatibility with ONNX files.
* Full model compilation for edge devices. 
* AMD computation compatibility.
* Cluster computation.
* Network analyzers.
* Transformers and convolutional layers.