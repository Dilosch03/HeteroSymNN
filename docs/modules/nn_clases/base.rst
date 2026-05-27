.. _base-network:

Base Network Architecture
=========================

The ``BaseNetwork`` class is the foundational chassis of HeteroSymNN. Every high-level builder (like :class:`~HeteroSymNN.Core.Nets.linear_net.MLP` , :class:`~HeteroSymNN.Core.Nets.linear_net.LinearNet`, and :class:`~HeteroSymNN.Core.Nets.linear_net.HeteroLinearNet`) is simply a structural wrapper that configures and instantiates this core engine.

If you are reading this page, you are likely doing one of three things:

1. **Building a Custom Topology:** Manually passing a mixed list of layer types to create an architecture that the standard wrappers don't support.
2. **Managing Hardware:** Taking manual control over CPU/GPU memory routing for performance benchmarking.
3. **Extending the Framework:** Writing a custom network wrapper class for a new research paradigm.

The Topology Sandbox
--------------------

The primary way to mix and match different Architecture Families is through the base class's ``network_structure`` initialization parameter. 

Instead of just passing a list of integers (node counts), the base class accepts tuples containing both the node count *and* the explicit Layer Class you want to use. This allows you to snap together a standard LinearNet layer with a future Recurrent or Evolutionary layer in the exact Network.

Hardware Routing & Memory Safety
--------------------------------

The ``BaseNetwork`` acts as the traffic controller for the entire graph. When you trigger the ``change_device("GPU")`` or ``change_device("CPU")`` methods at this level, the base class safely halts execution, flushes the memory queues, and recursively moves the underlying :type:`~HeteroSymNN.types.BackendArray` memory for every individual layer in the network.

State Extraction
----------------

Because HeteroSymNN networks can contain wildly different mathematical equations and dynamic constants per neuron, saving a model's state is complex. The base class handles this safely via ``get_config()``. It traverses through all its layers, extracts the deterministic blueprints and symbolic string configurations from every one, packing them into a single, serializable dictionary.

API Reference
-------------

.. automodule:: HeteroSymNN.Core.Nets.base_classes
   :members: BaseNetwork
   :undoc-members:
   :show-inheritance: