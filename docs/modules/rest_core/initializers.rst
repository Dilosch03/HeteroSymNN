.. _initializers:

Initializers
============

Proper weight initialization is mathematically critical for the stability of neural networks. This is especially true for HeteroSymNN's heterogeneous architectures, where different activation functions (e.g., periodic sine waves vs. linear ReLUs) scale gradients completely differently during the backward pass.

.. note::
   If you want to expand the types of initializers check the :ref:`developer_contract_initializers` section.

Standard Initializers
---------------------
The following initializers are pre-built and ready to use. Because they inherit from the base class, only their unique parameters and distinctly overridden methods are listed here.

.. automodule:: HeteroSymNN.Core.initializers
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Initializer


API Reference (Base Class)
--------------------------
.. autoclass:: HeteroSymNN.Core.initializers.Initializer
   :members:
   :undoc-members:

.. _developer_contract_initializers:

Developer Contract: Custom Initializers
---------------------------------------

If you are building a custom weight initialization algorithm (e.g., a custom variant of He or Glorot), your subclass must inherit from :class:`~HeteroSymNN.Core.initializers.Initializer` and fulfill the following operational lifecycles:

.. admonition:: Phase 0: Initialization
   :class: note
   
   If you override ``__init__`` to accept custom hyperparameters, you **must** call ``super().__init__()`` to ensure the base initializer properties are correctly set up before the generation phase.

.. admonition:: Phase 1: Generation Logic
   :class: note

   You must override the specific generation method required for your use case. The framework relies on the returned tensor strictly matching the layer's expected dimensions.

   * **``generate_from_distribution(self, shape, fan_in, fan_out)``**: Implement this if your initialization relies on statistical variance scaling (using the incoming/outgoing node counts).
   * **``generate_constant(self, shape, value)``**: Implement this for initializations that require the exact same value across the entire tensor.
   * **``generate_binary_mask(self, shape)``**: Implement this for the creation of topological boolean masks.
   * *Hardware Rule:* Initializers execute **strictly on the CPU**. All methods must generate and return standard ``np.ndarray`` (NumPy) objects. The calling Layer is responsible for pushing these arrays to the active device as :type:`~HeteroSymNN.types.BackendArray` structures.

.. admonition:: Phase 2: Serialization 
   :class: note

   To ensure your custom initialization strategy is reproducible and survives the ``.symnn`` save format, it must be capable of rebuilding its state.

   * **``get_config(self)``**: Must return a dictionary containing ``"class_name"`` and any specific hyperparameters required to reconstruct the object (e.g., a ``"gain"`` factor).
