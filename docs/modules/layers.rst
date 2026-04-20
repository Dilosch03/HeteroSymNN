.. _layer-classes:

Layer Classes
=============

In standard deep learning frameworks, layers are usually rigidly coupled to their specific mathematical operations. HeteroSymNN separates these concerns. Because the mathematical physics are handled dynamically by the JIT Compiler, **Layers in HeteroSymNN act purely as Physical Memory Managers.**

They are responsible for allocating the underlying :data:`~HeteroSymNN.types.BackendArray` buffers (in CPU RAM or GPU VRAM), securely storing the execution pointers to your compiled C++/CUDA kernels, and safely routing the forward and backward pass data streams between the hardware and the Python interface.

Why Read This Section?
----------------------

Most researchers will never need to instantiate these classes directly—they are automatically constructed, linked, and managed by the high-level Network Builders (like :class:`~HeteroSymNN.Core.Nets.Dense` or :class:`~HeteroSymNN.Core.Nets.HeteroDense`).

You should only consult this documentation if:

* **You are a Core Developer:** Looking to build a completely new layer topology (e.g., a dynamic Evolutionary layer or Recurrent layer) from scratch.
* **You are Debugging Memory:** Needing to understand how weights, biases, and connection masks are securely encapsulated and updated across CPU/GPU device boundaries.
* **You are Writing Custom Training Loops:** Needing to manually intercept the forward or backward pass tensors before they hit the JIT-compiled kernels.

Layer Implementations
---------------------

.. note::
    Currently the only implemented class is the linear layer but there are plans to implement recurrent layers and evolutionary layers.

* **LinearLayer:** The concrete implementation for standard feed-forward topological structures. It manages the standard ``(Nodes x Inputs)`` matrix structures and safely routes the outputs to the user's custom symbolic activation kernels.

.. toctree::
    :maxdepth: 1
    :caption: Implementations:

    layer_classes/linear_layer

Under the Hood: Core Architecture
---------------------------------

* **BaseLayer:** The abstract foundation. To prevent hardware-routing crashes and memory leaks, any custom layer topology you build *must* inherit from this class to guarantee compatibility with the ``BaseNetwork`` orchestrator.

.. toctree::
    :maxdepth: 1
    :caption: Core Architecture:

    layer_classes/general_layer