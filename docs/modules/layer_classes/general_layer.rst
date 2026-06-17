.. _base-layer:

BaseLayer
=========

The Concept
-----------
The ``BaseLayer`` is the abstract topological foundation of HeteroSymNN. It establishes a strict physical contract that guarantees the central :customref:`BaseNetwork <base-network>` orchestrator can safely calculate dimensions, route gradients, and move memory arrays between the CPU and GPU without knowing the specific internal mechanics of the layer.

When to Use
-----------
You should only interact with this class if you are **extending the framework**. If you want to create a completely new structural topology (e.g., a Convolutional layer, a Recurrent layer, or a dynamically mutating EvoNet layer), you must inherit from ``BaseLayer`` to ensure your new physics engine integrates seamlessly with the rest of the ecosystem.
:ref:`developer_contract_layers`

API Reference
-------------

.. autoclass:: HeteroSymNN.Core.layers.BaseLayer
   :members:
   :undoc-members:

.. _developer_contract_layers:

Developer Contract: Custom Layers
---------------------------------

If you are building a custom layer topology, your subclass must inherit from :class:`~HeteroSymNN.Core.layers.BaseLayer` and fulfill the following operational lifecycles to prevent hardware-routing crashes or memory leaks:

.. admonition:: Phase 1: Hardware & Memory Allocation
   :class: note

   In your ``__init__`` method, you must allocate any structural parameters (weights, biases, recurrent state matrices). 
   
   * **Initialization**: You **must** call ``super().__init__(num_inputs, layer_configuration, batch_size, gpu_id)`` to initialize the base architecture.
   * **Hardware Rule:** You must respect the layer's current hardware state. Use ``self._CALCULATION_MANAGER`` to allocate arrays, ensuring they are placed in CPU RAM or GPU VRAM appropriately based on the user's global settings.
   * **Masking Rule:** The structural masks (e.g., connection topologies) and mathematical parameters (weights) MUST be saved as separate arrays. Mask parameters must ALWAYS be linear_net arrays, not sparse matrix structures.

.. admonition:: Phase 2: Mathematical Routing (Forward/Backward)
   :class: note

   You must override the primary execution hooks. The JIT compiler handles the symbolic activation math, but your layer is responsible for the macroscopic data flow.

   * **``forward(self, inputs)``**: Must calculate the pre-activation outputs (usually stored in ``self.z``) and route them to the activation functions, returning a strictly shaped :type:`~HeteroSymNN.types.BackendArray`.
   * **``backward(self, next_layer_errors)``**: Must calculate the local gradient (``self.delta``) and correctly route the error backwards through the topological structure.
   * **Optimization Hooks**: For backpropagation to successfully update the layer, you must also define and expose the ``working_parameters``, ``working_gradients``, and ``working_masks`` properties so the framework's optimizers can read and safely mutate your custom structures.

.. admonition:: Phase 3: State Serialization
   :class: note

   To ensure your custom layer topology survives the ``.symnn`` save format, it must be capable of extracting its deterministic blueprints.

   * **``get_config(self)``**: Must return a dictionary containing the layer's symbolic string configurations, dimensions, and any custom hyperparameters required to reconstruct the object (e.g., ``"recurrent_dropout_rate"``).
   * **``get_parameters(self)``**: Must extract and return a dictionary mapping parameter names to their current physical arrays (e.g., weights and biases).
   * **``set_parameters(self, params)``**: Must correctly inject a dictionary of loaded parameter arrays back into your hardware-allocated properties.
