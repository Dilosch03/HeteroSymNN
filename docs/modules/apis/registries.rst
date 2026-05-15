.. _registries:

Component Registries
====================

The ``registries`` module solves a fundamental problem in dynamic framework architecture: **Safe Deserialization.**

When you save a model to a ``.symnn`` archive, the system records the *string names* of your network topology, layers, loss functions, and optimizers. Upon loading, the framework needs a way to translate those strings back into executable Python classes without using highly insecure ``eval()`` operations. 

The internal ``_Registry`` acts as this translation layer. Upon initialization, it automatically scans the Core modules and dynamically builds a dictionary mapping every string identifier to its actual class object in memory.

When to Use
-----------
Because the registry automatically discovers standard HeteroSymNN components, **you only need to interact with this module if you are loading a saved model that contains custom-built extensions.** If you have built a custom Network architecture or a custom Data Transformer, you must manually inject it into the global registry *before* calling ``load_model()``, otherwise the framework will not know how to rebuild your custom objects.

Code Example
------------
You do not instantiate a new registry. Instead, you import the global ``registry`` singleton and use its injection methods.

.. code-block:: python
      
    from HeteroSymNN.API.registries import registry
    from HeteroSymNN.API.wrappers import Wrapper
    from my_custom_module import MyBespokeNetwork, MyLogTransformer

    # 1. Register your custom classes so they can survive a load cycle
    registry.add_net(MyBespokeNetwork)
    registry.add_data_transformer(MyLogTransformer)

    # 2. Now you can safely load a .symnn file that uses these custom classes
    agent = Wrapper.load_model("my_custom_agent.symnn")

API Reference
-------------

.. currentmodule:: HeteroSymNN.API.registries

.. autodata:: registry
   :annotation: = Global instance of _Registry

   The active singleton managing all class mappings. You can inspect the currently registered components by accessing its read-only dictionary properties:

   * ``registry._net_map``: Available network architectures.
   * ``registry._transformers_map``: Available data scaling transformers.
   * ``registry._loss_fn_map``: Available symbolic loss functions.
   * ``registry._optimizers_map``: Available optimization algorithms.
   * ``registry._initializers_map``: Available weight initializers.
   * ``registry._layers_map``: Available layer structures.

.. autoclass:: _Registry
   :members: add_net, add_data_transformer,add_loss_func,add_optimizer,add_initializer,add_layer
   :undoc-members:

   .. note::
      **Type Safety:** The registration methods enforce strict typing. If you attempt to register a network that does not inherit from :class:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork`, or a transformer that does not inherit from :class:`~HeteroSymNN.API.data_transformers.DataTransformer`, the registry will safely reject it and raise a :exc:`~HeteroSymNN.exceptions.DataTypeError`.