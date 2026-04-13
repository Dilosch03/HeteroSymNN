.. _registries:

Component Registries
====================

The ``registries`` module solves a fundamental problem in dynamic framework architecture: **Safe Deserialization.**

When you save a model to a ``.symnn`` archive, the system records the *string names* of your network topology, layers, loss functions, and optimizers. Upon loading, the framework needs a way to translate those strings back into executable Python classes without using highly insecure ``eval()`` operations. 

The ``_Registry`` acts as the translation layer, dynamically mapping string identifiers to their actual class objects in memory.

.. currentmodule:: HeteroSymNN.API.registries

The Global Registry
-------------------

.. autodata:: registry
   :annotation: = Global instance of _Registry

   The active singleton managing all class mappings. Upon initialization, it automatically traverses the ``Core`` modules (losses, optimizers, initializers, layers, Nets) and dynamically builds the translation dictionaries.

Available Component Maps
------------------------

You can inspect the currently registered components by accessing the read-only dictionary properties on the global ``registry`` object:

* ``registry.net_map``: Available network architectures.
* ``registry.transformers_map``: Available data scaling transformers.
* ``registry.loss_fn_map``: Available symbolic loss functions.
* ``registry.optimizers_map``: Available optimization algorithms.
* ``registry.initializers_map``: Available weight initializers.
* ``registry.layers_map``: Available layer structures.

Adding Custom Components
------------------------

Because the registry automatically discovers internal HeteroSymNN components, you only need to manually register **external, custom-built** architectures or transformers so the ``Wrapper`` knows how to load them from a saved file.

.. autoclass:: _Registry
   :members:
   :undoc-members:

   .. note::
      **Type Safety:** The registration methods enforce strict typing. If you attempt to register a network that does not inherit from :class:`~HeteroSymNN.Core.Nets.BaseNetwork`, or a transformer that does not inherit from :class:`~HeteroSymNN.API.utilities.DataTransformer`, the registry will raise a :data:`~HeteroSymNN.exceptions.DataTypeError`.

   **Example Usage:**

   .. code-block:: python
      
      from HeteroSymNN.API.registries import registry
      from my_custom_module import MyBespokeNetwork, MyLogTransformer

      # Register your custom classes so they can survive a save/load cycle
      registry.add_net(MyBespokeNetwork)
      registry.add_data_transformer(MyLogTransformer)