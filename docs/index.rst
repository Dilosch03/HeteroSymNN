Welcome to HeteroSymNN
======================

**HeteroSymNN** is a symbolic JIT-Compiled Deep Learning framework designed specifically for Heterogeneous Neural Networks.

Unlike standard deep learning frameworks that optimize for homogeneous layer matrices, HeteroSymNN utilizes a Symbolic JIT Compiler to generate dynamically fused kernels at runtime. This architecture allows every single neuron within a layer to possess a distinct, custom mathematical activation function (e.g., ``sin(x)``, ``tanh(x)``, ``alpha * x + beta``) with **zero computational overhead** during the forward and backward passes.

.. note::
   This framework is explicitly tailored for Neuroevolution (NEAT architectures), Advanced Control Systems, and Scientific Machine Learning (SciML) where architectural flexibility supersedes rigid matrix multiplications.

The Architecture: How It Works
------------------------------

HeteroSymNN operates as a **Differentiable Compiler** rather than a standard tensor operations library. The pipeline consists of four distinct phases:

1. **Parse**: Accepts human-readable mathematical strings (e.g., ``"alpha * sin(x)"``) and parses them into abstract syntax trees using SymPy.
2. **Derive**: Automatically computes the exact symbolic derivative for backpropagation, eliminating the need for autograd tracking at runtime.
3. **Compile**: Generates low-level C++ or CUDA code Just-In-Time. It creates optimized ``switch`` statements to handle distinct neuron instructions concurrently.
4. **Fuse**: Fuses all memory access and execution logic into a single hardware kernel launch, drastically reducing memory bandwidth bottlenecks.

Installation
------------

HeteroSymNN defaults to a highly portable CPU JIT/Python execution mode:

.. code-block:: bash

   pip install heterosymnn

For high-performance GPU acceleration (Requires NVIDIA Drivers & CuPy):

.. code-block:: bash

   pip install heterosymnn[gpu]

:doc:`Read the full installation guide <installation>` for compiler prerequisites and backend details.

Quickstart: The "Cocktail" Layer
--------------------------------

Here is a brief example of instantiating a network that seamlessly mixes periodic (Sine) and linear features in the exact same layer—a task that is notoriously inefficient in standard frameworks.

.. code-block:: python

    from HeteroSymNN.Core.Nets import Dense
    from HeteroSymNN.API import Wrapper

    # Define a network with 10 inputs, two hidden layers of 25 nodes, and 1 output.
    # The activation config assigns specific symbolic functions to the layers.
    model = Dense(
        nodes_structure=[10, 25, 25, 1],
        activation_config=["sin(num)", "num", ("tanh(num)*a", {"a": 2.0})]
    )

    # Wrap the model for Scikit-Learn style training (Regression mode)
    agent = Wrapper(model, work_type="reg")
    agent.fit(X_train, y_train, epochs=100)

:doc:`Explore the Quickstart <quickstart>` for a deeper dive into node configurations.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart

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