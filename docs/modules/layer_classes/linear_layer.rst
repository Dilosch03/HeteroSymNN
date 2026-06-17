.. _linear-layer:

LinearLayer (The Workhorse)
===========================

The ``LinearLayer`` is the concrete implementation of a standard, fully-connected bipartite graph. It acts as the physical memory manager for a 2D weight matrix, a 1D bias vector, and a boolean connection mask. It handles the linear_net matrix multiplication ``(X @ W^T + B)`` and routes the resulting output stream to the JIT-compiled symbolic activation kernels.

When to Use
-----------
This is the default topological layer used automatically by the entire Feed-Forward family (:customref:`MLP <mlp-network>`, :customref:`LinearNet <dense-network>`, :customref:`HeteroLinearNet <heterodense-network>`). 

You generally do not need to use this class manually. You should only pass the class definition directly if you are using the :customref:`BaseNetwork <base-network>` to manually build a custom "Frankenstein" architecture that mixes different topological families.

Code Example
------------
If you are manually constructing a custom topology, you can initialize a ``LinearLayer`` by explicitly defining its physical dimensions and its symbolic configuration.

.. code-block:: python

    from HeteroSymNN.Core.layers import LinearLayer
    from HeteroSymNN.Core.Nets.initializers import HeNormal

    # 1. Define the symbolic physics for the 25 nodes
    node_configs = [("sin(num*a)", {"a": 1.5})] * 25

    # 2. Instantiate the physical layer (expects 10 inputs)
    custom_layer = LinearLayer(
        num_inputs=10,
        layer_configuration=(node_configs, HeNormal())
    )

API Reference
-------------

.. autoclass:: HeteroSymNN.Core.layers.LinearLayer
   :members:
   :undoc-members:
   :show-inheritance: