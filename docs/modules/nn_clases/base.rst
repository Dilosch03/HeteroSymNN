.. _base-network:

Base Network Architecture
=========================

The ``BaseNetwork`` class is the foundational chassis of HeteroSymNN. Every high-level builder (like :customref:`MLP <mlp-network>` , :customref:`LinearNet <dense-network>`, and :customref:`HeteroLinearNet <heterodense-network>`) is simply a structural wrapper that configures and instantiates this core engine.

If you are reading this page, you are likely doing one of three things:

1. **Building a Custom Topology:** Manually passing a mixed list of layer types to create an architecture that the standard wrappers don't support.
2. **Managing Hardware:** Taking manual control over CPU/GPU memory routing for performance benchmarking.
3. **Extending the Framework:** Writing a custom network wrapper class for a new research paradigm.

The Topology Sandbox
--------------------

The primary way to mix and match different Architecture Families is through the base class's ``network_structure`` initialization parameter. 

The base class accepts a list of the explicit Layer Class you want to use. This allows you to snap together a standard LinearNet layer with a future Recurrent, Evolutionary or convolutional layer in the exact Network.

Hardware Routing & Memory Safety
--------------------------------

The ``BaseNetwork`` acts as the traffic controller for the entire graph. When you trigger the ``to("host")`` or ``to("device")`` methods at this level, the base class safely halts execution, flushes the memory queues, and recursively moves the underlying :type:`~HeteroSymNN.types.BackendArray` memory for every individual layer in the network.

State Serialization & Persistence
---------------------------------

Because HeteroSymNN networks can contain wildly different mathematical equations and dynamic constants per neuron, saving a model's state is complex. The base class provides built-in methods to save and load models directly without wrapping them:

* **Saving:** You can call :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.save_model` directly on any network instance to save its architecture, parameter weights, and optimizer states.
* **Loading as a new instance:** You can call the class method :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.load_model` to load and instantiate a saved network from a file.
* **Loading state into an existing instance:** You can call :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.load_state` to restore parameter weights and optimizer states into an existing model instance.

API Reference
-------------

.. automodule:: HeteroSymNN.Core.Nets.base_classes
   :members: BaseNetwork
   :undoc-members:
   :show-inheritance: