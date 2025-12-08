Welcome to HeteroSymNN's documentation!
=======================================

**HeteroSymNN** is a symbolic JIT-Compiled Deep Learning framework for Heterogeneous Neural Networks.

Unlike standard frameworks that optimize for homogeneous layers, HeteroSymNN uses a Symbolic JIT Compiler to generate fused kernels at runtime. This allows every single neuron in a layer to have a distinct, custom mathematical activation function (e.g., ``sin(x)``, ``tanh(x)``, ``alpha * x + beta``) with zero computational overhead.

.. note::
   This project is tailored for Neuroevolution (NEAT), Control Systems, and Scientific Machine Learning.

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
   modules/jit
   modules/backends

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`