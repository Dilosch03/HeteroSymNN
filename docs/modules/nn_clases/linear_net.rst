.. _dense-network:

LinearNet (The Mixed-Activation Network)
============================

The ``LinearNet`` class expands the capabilities of the :customref:`MLP <mlp-network>` by allowing you to define completely different activation functions on a per-layer basis. 

When to Use
-----------
Use this class when your research requires macroscopic heterogeneity. It operates very similarly to the standard sequential classes found in PyTorch and JAX, but maintains the powerful ability to easily define custom, symbolic mathematical functions for each distinct layer without writing custom boilerplate code.

Code Example
------------
Instantiating the class stays relatively simple, just changing the activation parameter to require a list of the used functions.

.. code-block:: python

    from HeteroSymNN.Core.Nets import LinearNet
    from HeteroSymNN.API import Wrapper

    # 10 inputs, two hidden layers (25 nodes each), 1 output
    baseline_model = LinearNet(
        nodes_structure=[10, 25, 25, 1],
        activation_config=["sin(num)", "num", ("tanh(num)*a", {"a": 2.0})],
        learning_rate=0.01,
        num_training_iter = 50
    )

    # 2. Wrap and train
    agent = Wrapper(baseline_model, work_type="class")
    agent.fit(X_train, y_train)

API Reference
-------------

.. autoclass:: HeteroSymNN.Core.Nets.LinearNet
   :members:
   :undoc-members:
   :show-inheritance: