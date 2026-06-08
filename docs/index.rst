Welcome to HeteroSymNN
======================

**HeteroSymNN** is a symbolic JIT-Compiled Deep Learning framework designed specifically for Heterogeneous Neural Networks.

Unlike standard deep learning frameworks that optimize for homogeneous layer matrices, HeteroSymNN utilizes a Symbolic JIT Compiler to generate dynamically fused kernels at runtime. This architecture allows for mathematical flexibility—from assigning custom symbolic activation functions (e.g., ``sin(num)``, ``tanh(num)``, ``alpha * num + beta``) per layer, all the way down to customizing individual neurons, with **minimal computational overhead**.

.. note::
   This framework is explicitly tailored for Advanced Control Systems and Scientific Machine Learning (SciML) where architectural flexibility supersedes rigid matrix multiplications.

Installation
------------

HeteroSymNN defaults to highly portable CPU operations using numpy:

.. code-block:: bash

   pip install heterosymnn

But can also be installed with high-performance GPU acceleration using CuPy (Requires NVIDIA Drivers):

.. code-block:: bash

   pip install heterosymnn[gpu]

:doc:`Read the full installation guide <installation>` for compiler prerequisites and backend details.

Quickstart: The Mixed-Activation Network
----------------------------------

Here is a brief example of how easy it is to instantiate a network that seamlessly mixes different symbolic features across its layers. Standard frameworks require custom classes for this; HeteroSymNN just requires human-readable strings.

.. code-block:: python

    from HeteroSymNN.Core.Nets import LinearNet
    from HeteroSymNN.API import Wrapper

    # Define a network with 10 inputs, two hidden layers of 25 nodes, and 1 output.
    # The activation_config assigns a specific symbolic function to each layer.
    model = LinearNet(
        nodes_structure=[10, 25, 25, 1],
        activation_config=["sin(num)", "num", ("tanh(num)*a", {"a": 2.0})]
    )

    # Wrap the model for Scikit-Learn style training (Regression mode)
    agent = Wrapper(model, work_type="reg")
    agent.fit(X_train, y_train, epochs=100)

:doc:`Explore the Quickstart <quickstart>` for a deeper dive into node-level configurations and zero recompilation constant.

The Architecture: How It Works
------------------------------

HeteroSymNN operates as a **Differentiable Compiler** rather than a standard tensor operations library. The pipeline consists of four distinct phases:

1. **Parse**: Accepts human-readable mathematical strings (e.g., ``"alpha * sin(num)"``) and parses and formats them into a structure for execution using SymPy.
2. **Derive**: Automatically computes the exact symbolic derivative for backpropagation, eliminating the need for autograd tracking at runtime.
3. **Compile**: Generates low-level C++ or CUDA code Just-In-Time. 
4. **Fuse**: Fuses all memory access and execution logic into a single hardware kernel launch, drastically reducing memory bandwidth bottlenecks.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart
   custom_training_loop
   cli

.. toctree::
   :maxdepth: 1
   :caption: Project Information:

   benchmarks
   roadmap
   changelog
   contribution

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules/api
   modules/core
   modules/types
   modules/jit
   modules/backend
   modules/config
   modules/exceptions