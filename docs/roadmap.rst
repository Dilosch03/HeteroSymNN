.. _roadmap:

Development Roadmap
===================

HeteroSymNN is currently in active beta. The architecture is evolving towards a highly secure, modular, and extensible ecosystem.

Current Phase: v0.3.x (API Stabilization)
-----------------------------------------
* [Completed] Restructure of the project for scalability and modularity.

Upcoming: v0.4.0 (Security & Persistence)
-----------------------------------------
* **Lexical Analyzer & Global Registry**: Moving constant extraction and string expansion into the SymbolicJITCompiler.
    * Implementing the Global Function Registry (settings.register_shortcut()) for alias expansion before SymPy compilation.

* **API Quality of Life** (SSOT Enforcement):
    * Deprecating training_mode and strictly enforcing batch_size as the Single Source of Truth.
    * Deprecating the num_nodes parameter in HeteroDense and inferring it dynamically from len(detailed_activations).
    * Converting magic routing strings (like "GPU_CUDA") into strict String-Enums to enable IDE autocomplete and prevent typos.
    * Adding the available_computational_methods property to GlobalSettings.

* **Asynchronous Data Bus**: Building the DatasetStreamer to feed the GPU in chunks without overflowing CPU RAM.
* **JIT Compiler Evolution**: Implementing AST Safeguards and Dynamic Loop Unrolling for recursive math.
    * Making a modular architecture for expansion of the capabilities.
* **Modular Evaluation**: Moving the metrics out of the wrapper so they can be inserted as dependencies (Custom R2, F1, etc.).
* **Unified Training Controller**: Wrapping Early Stopping and Schedulers around the Optimizer, and building the InitializerManager (LSUV/statistical initialization).
* **Data type runtime separation**: Make it so you can train different networks with different data types (one on float32 other on int16).
  
Upcoming: v0.5.0 (Advanced Routing & Topologies)
------------------------------------------------
* **Recurrent Networks (RNNs)**: Implementing Backpropagation Through Time (BPTT) math natively on the Heavy Forge (C++ JIT) to verify gradients perfectly.
* **Internal Normalization**: Adding BatchNorm and LayerNorm natively into the JIT pipeline to prevent exploding gradients in deeper topologies.
* **Memory Orchestrators**: Advanced automated RAM/VRAM paging.
* **Heterogeneous optimizers**: Adding the ability to set per neuron the type of optimizer is going to be used to update it.
* **Heterogeneous initializer**: Adding the ability to set depending on the data type is been ask use different initializer function.

Upcoming: v0.6.0 (Dual-Engine Architecture)
--------------------------------------------
* CUDA VM: Building the Python Bytecode Assembler and a Cuda Virtual Machine to execute opcodes directly in VRAM without invoking the nvcc compiler.
* Decided JIT:Implementing the heuristic algorithm to decide if it's going to use the interpreter or the Jit compiled kernels during training.

Future: v0.7.0+
----------------
* Evolutionary Network.
    * Buffer and Flush Architecture: Implementing the `_is_dirty` flag. Mutating the entire next generation's math in CPU NumPy arrays, and flushing it to the GPU in a single massive PCIe payload.
* N-Dimentional support: Upgrading the kernels to support the Z-axis (Population Index) and executing evaluations via Batched Matrix Multiplication (BMM).
* Model Quantization.
* Compatibility with ONNX files.
* Full model compilation for edge devices. 
* AMD computation compatibility.
* Cluster computation.
* Network analyzers.
* Transformers and convolutional layers.