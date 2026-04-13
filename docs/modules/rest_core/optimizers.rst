.. _optimizers:

Optimizers
==========

Optimizers are responsible for applying the gradients—derived mathematically by the Symbolic JIT Compiler during the backward pass—to the underlying weights and biases of the network's layers.

.. note::
   **Extending the Framework:** Are you looking to build your own custom optimization algorithm? Jump straight to the :ref:`developer_contract_optimizers`.

.. currentmodule:: HeteroSymNN.Core.optimizers

Base Optimizer
--------------

.. autoclass:: Optimizer
   :members:
   :undoc-members:

   The structural foundation for all optimization algorithms in the framework. It defines the universal contract for applying calculated gradients to layer memory buffers. All optimizers below inherit from this template.

Standard Optimizers
-------------------

The following optimization algorithms are pre-built and ready to use. Only their specific parameters and overridden execution steps are detailed here.

.. automodule:: HeteroSymNN.Core.optimizers
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Optimizer


.. _developer_contract_optimizers:

Developer Contract: Extending Optimizers
-----------------------------------------

If you are building a custom optimization algorithm (e.g., a custom variant of Adam or RMSprop), your subclass must inherit from :class:`~HeteroSymNN.Core.optimizers.Optimizer` and fulfill three distinct operational lifecycles to ensure stability within the framework.

.. admonition:: Phase 1: The Execution Cycle (Math)
   :class: note

   * **``step(layers)`` or ``update_params(layers)``**: This is the core mathematical engine. It must iterate through the provided list of :class:`~HeteroSymNN.Core.layers.BaseLayer` instances. For each layer, it must fetch the exact gradients (using the layer's internal memory methods), apply your mathematical scaling (e.g., momentum, learning rate), and safely update the primary weight/bias matrices.

.. admonition:: Phase 2: Serialization & State Management
   :class: note

   To allow users to save and resume training using the ``.symnn`` format, your optimizer must separate its static parameters from its dynamic memory.

   * **``get_config()``**: Must return a dictionary of your static hyperparameters (e.g., ``learning_rate``, ``beta1``). It must include ``"class_name"`` so the Registry knows how to rebuild it.
   * **``get_state()``**: Must return a dictionary containing the dynamic training state (e.g., current iteration step ``t``, or accumulated momentum tensors).
   * **``set_state(state)``**: Must accept the dictionary from ``get_state()`` and successfully restore the internal momentum tensors to resume training exactly where it left off.

.. admonition:: Phase 3: Hardware Routing (Advanced)
   :class: warning

   If your custom optimizer utilizes internal state arrays (like velocity or running averages), you must implement hardware routing to prevent memory desynchronization when a user dynamically switches from CPU to GPU.

   * **``_change_COMPUTATIONAL_METHOD(new_method, gpu_id)``**: You must intercept this call to manually migrate any of your internal state arrays from Host RAM (NumPy) to Device VRAM (CuPy), or vice versa, ensuring they match the active :data:`~HeteroSymNN.types.BackendArray` type of the network's layers.