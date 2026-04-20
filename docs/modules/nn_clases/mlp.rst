.. _mlp-network:

MLP (Multi-Layer Perceptron)
============================

The ``MLP`` class provides a streamlined builder for standard, homogeneous feed-forward neural networks. In this architecture, all hidden layers share the exact same mathematical activation function, mirroring the behavior of traditional deep learning models.

When to Use
-----------
Use this class to establish a rapid **standard baseline** for your experiments. Before introducing complex, per-node symbolic math with advanced builders, the ``MLP`` allows you to quickly verify that your data pipeline, training loops and weird math functions are functioning correctly using a proven, homogeneous topology.

Code Example
------------
Building a baseline model requires minimal configuration. By default, hidden layers will use ``"relu"`` and the output layer will use ``"num"`` (a linear pass-through).

.. code-block:: python

    from HeteroSymNN.Core.Nets import MLP
    from HeteroSymNN.API import Wrapper

    # 1. Define a standard baseline network
    # 10 inputs, two hidden layers (25 nodes each), 1 output
    baseline_model = MLP(
        nodes_structure=[10, 25, 25, 1],
        activation="relu",
        learning_rate=0.01
    )

    # 2. Wrap and train
    agent = Wrapper(baseline_model, work_type="class")
    agent.fit(X_train, y_train, epochs=50)

API Reference
-------------

.. autoclass:: HeteroSymNN.Core.Nets.MLP
   :members:
   :undoc-members:
   :show-inheritance: