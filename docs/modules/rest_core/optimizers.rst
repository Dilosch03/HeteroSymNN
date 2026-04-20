.. _optimizers:

Optimizers
==========

Optimizers are responsible for applying the gradients—derived mathematically by the Symbolic JIT Compiler during the backward pass—to the underlying weights and physical biases of the network's layers.

.. note::
   If you want to expand the types of optimizers check the :ref:`developer_contract_optimizers` section.

Standard Optimizers
-------------------
The following optimization algorithms are pre-built and ready to use. Only their specific hyperparameters and overridden execution steps are detailed here.

.. automodule:: HeteroSymNN.Core.optimizers
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Optimizer

API Reference (Base Class)
--------------------------
.. autoclass:: HeteroSymNN.Core.optimizers.Optimizer
   :members:
   :undoc-members:

.. _developer_contract_optimizers:

Developer Contract: Extending Optimizers
----------------------------------------

If you are building a custom optimization algorithm (e.g., a custom variant of Adam or RMSprop), your subclass must inherit from :class:`~HeteroSymNN.Core.optimizers.Optimizer` and fulfill three distinct operational lifecycles to ensure stability within the framework:

.. admonition:: Phase 1: Gradient Application
   :class: note

   You must override the primary update loop. The framework will pass you the physical layer arrays and their calculated gradients. You must apply your mathematical scaling (e.g., momentum, velocity) and safely update the primary parameter matrices in-place.

.. admonition:: Phase 2: Serialization & State Management
   :class: note

   To allow users to save and resume training using the ``.symnn`` format, your optimizer must separate its static configurations from its dynamic memory.

   * **``get_config()``**: Must return a dictionary of static hyperparameters (e.g., ``learning_rate``). It must include ``"class_name"``.
   * **``get_state()``**: Must return a dictionary containing the dynamic training state (e.g., current iteration step ``t``, or accumulated momentum tensors).
   * **``set_state(state)``**: Must accept the dictionary from ``get_state()`` and successfully restore the internal tensors to resume training exactly where it left off.

.. admonition:: Phase 3: Hardware Routing (Advanced)
   :class: warning

   If your custom optimizer utilizes internal state arrays (like velocity or running averages), you must implement hardware routing to prevent memory desynchronization when a user dynamically switches between CPU and GPU.

   * **``_change_COMPUTATIONAL_METHOD(new_method, gpu_id)``**: You must intercept this call to manually migrate any of your internal state arrays from Host RAM (NumPy) to Device VRAM (CuPy), or vice versa.
