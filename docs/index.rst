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
   
.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   usage

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