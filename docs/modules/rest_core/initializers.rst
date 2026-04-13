.. _initializers:

Initializers
============

Proper weight initialization is mathematically critical for the stability of neural networks, and this is especially true for heterogeneous architectures where different activation functions (e.g., periodic vs. linear) may scale gradients differently during the backward pass.

.. note::
   **Extending the Framework:** Need to implement a custom statistical distribution for your weights? Jump straight to the :ref:`developer_contract_initializers`.

.. currentmodule:: HeteroSymNN.Core.initializers

Base Initializer
----------------

.. autoclass:: Initializer
   :members:
   :undoc-members:

   The structural foundation for initializing weight and bias matrices within HeteroSymNN layers. All standard initializers below inherit the base properties and methods defined here.

Standard Initializers
---------------------

The following initializers are pre-built and ready to use. Because they inherit from the base class above, only their unique parameters and distinctly overridden methods are listed here.

.. automodule:: HeteroSymNN.Core.initializers
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Initializer

.. _developer_contract_initializers:

Developer Contract: Custom Initializers
------------------------------------------

If you are building a custom weight initialization algorithm (e.g., a custom variant of He or Glorot), your subclass must inherit from :class:`~HeteroSymNN.Core.initializers.Initializer` or :class:`~HeteroSymNN.Core.initializers.BaseInitializer` and fulfill the following two operational lifecycles.

.. admonition:: Phase 1: Generation Logic
   :class: note

   You must override ``generate_from_distribution()`` and depending on the inherit can override any of the 2 remaining methods depending on your use case. The framework relies on the returned tensor strictly matching the layer's expected dimensions.

   * **``generate_from_distribution(self, shape, fan_in, fan_out)``**: Implement this if your initialization relies on statistical variance scaling (using the incoming/outgoing node counts).
   * **``generate_constant(self, shape, value)``**: Implement this for initializations that require the same value across the tensor.
   * **``generate_binary_mask(self, shape)``**: Implement this for the creation of binary masks.
   * *Requirement:* All methods must return a correctly shaped :data:`~HeteroSymNN.types.BackendArray` (e.g., ``np.ndarray`` cast to the correct floating precision).

.. admonition:: Phase 2: Serialization 
   :class: note

   To ensure your custom initialization strategy is reproducible and survives the ``.symnn`` save format, it must be capable of rebuilding its state.

   * **``get_config(self)``**: Must return a dictionary containing ``"class_name"`` and any specific hyperparameters required to reconstruct the object (e.g., a ``"gain"`` factor or a ``"connection_density"`` mask).