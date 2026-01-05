Welcome to HeteroSymNN's documentation!
=======================================

**HeteroSymNN** is a symbolic JIT-Compiled Deep Learning framework for Heterogeneous Neural Networks.

Unlike standard frameworks that optimize for homogeneous layers, HeteroSymNN uses a Symbolic JIT Compiler to generate fused kernels at runtime. This allows every single neuron in a layer to have a distinct, custom mathematical activation function (e.g., ``sin(x)``, ``tanh(x)``, ``alpha * x + beta``) with zero computational overhead.

.. note::
   This project is tailored for Neuroevolution (NEAT), Control Systems, and Scientific Machine Learning.

How It Works
------------

HeteroSymNN acts as a **Differentiable Compiler**:

1. **Parse**: Accepts mathematical strings (e.g., ``"alpha * sin(x)"``) and parses them using SymPy.
2. **Derive**: Automatically calculates the symbolic derivative for backpropagation.
3. **Compile**: Generates C++ or CUDA code at runtime, creating a ``switch`` statement for distinct neuron instructions.
4. **Fuse**: Fuses memory access into a single kernel launch.

Installation
------------

Standard installation (CPU Python/JIT):

.. code-block:: bash

    pip install heterosymnn

To enable high-performance GPU acceleration (requires NVIDIA Drivers & CuPy):

.. code-block:: bash

    pip install heterosymnn[gpu]

:doc:`For more details, see the installation guide</installation>`

-------------------------------------------

Quickstart: The "Cocktail" Layer
--------------------------------

Here is a 5-line example of creating a network that mixes periodic (Sin) and linear features in the same layer.

.. code-block:: python

    from heterosymnn.core import FlexibleNN
    from heterosymnn.api import Wraper

    # Define a layer with 25 Sine neurons and 25 Linear neurons
    model = FlexibleNN(
        input_size=10,
        layer_specs=[([(25, "sin(num)"), (25, "num")])]
    )

    # Train using Scikit-Learn style API
    agent = Wraper(model)
    agent.fit(X_train, y_train, epochs=100)

:doc:`For more examples, see the quickstart guide</usages>`

-------------------

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   usages

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules/api
   modules/core
   modules/types
   modules/jit
   modules/backend

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`