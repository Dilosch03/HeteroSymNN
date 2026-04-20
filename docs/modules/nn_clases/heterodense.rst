.. _heterodense-network:

HeteroDense (Node-Level Customization)
============================

The ``HeteroDense`` class represents the microscopic frontier of the Feed-Forward family. It gives you the unprecedented ability to assign a completely unique activation function and dynamic symbolic constants to *every individual neuron* in the network, fusing these distinct instructions into a single execution pass.

When to Use
-----------
Use this class when your research demands absolute, microscopic control. It provides the full heterogeneity perfect for Physics-Informed Neural Networks (PINNs), advanced Scientific Machine Learning (SciML) models, and highly specialized architectures where individual nodes must obey entirely different mathematical laws or physics constraints.

Code Example
------------
Instantiating this class requires that you assign a detailed list of activation functions to each layer using the :data:`~HeteroSymNN.types.NodeConfig` format.

.. code-block:: python

    from HeteroSymNN.Core.Nets.dense import HeteroDense
    from HeteroSymNN.API import Wrapper

    # 4 inputs, one hidden layer (6 nodes), 3 outputs
    layer0_config = [
        ("tanh(num*a)", {"a": 1.6}), 
        ("sigmoid", {}), 
        ("sigmoid", {}), 
        ("sigmoid", {}), 
        ("tanh(num*a)", {"a": 3.8}), 
        ("tanh(num*a)", {"a": -48})
    ]
    outputlayer_config = [("relu", {}), ("sin(num)", {}), ("tanh(num)", {})]
    
    baseline_model = HeteroDense(
        nodes_structure=[4, 6, 3],
        detailed_activations=[layer0_config, outputlayer_config]
    )

    # Then wrap and train
    agent = Wrapper(baseline_model, work_type="class")
    agent.fit(X_train, y_train)

API Reference
-------------

.. autoclass:: HeteroSymNN.Core.Nets.MLP
   :members:
   :undoc-members:
   :show-inheritance: