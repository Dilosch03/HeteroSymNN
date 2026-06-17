.. _contributing:

Contributing to HeteroSymNN
===========================

We welcome contributions from the community! Because HeteroSymNN operates at the complex intersection of hardware memory management and JIT compilation, we ask that all contributors adhere to some structural philosophies.

The Separation of Concerns
--------------------------
Before submitting a Pull Request, ensure your code respects the framework's strict architectural boundaries:

1. **Layers are strictly Memory Managers.** Do not write mathematical physics or activation logic inside a Layer class. Layers only manage :type:`~HeteroSymNN.types.BackendArray` allocations and data routing.
2. **Math belongs to the JIT.** All mathematical operations, gradients, and custom loss functions should be parsed symbolically through the :customref:`SymbolicJITCompiler <jit>`.
3. **Respect the Fallback.** Any new feature must work on all three backends (``GPU_CUDA``, ``CPU_JIT``, and ``CPU_PYTHON``). Do not write CuPy-exclusive features without providing a NumPy equivalent.

Developer Contracts
-------------------
If you are adding a new component, please review the relevant Developer Contract in the documentation to ensure you implement the correct state serialization (``get_config()`` / ``set_state()``) methods:

* :ref:`developer_contract_layers`
* :ref:`developer_contract_initializers`
* :ref:`developer_contract_losses`
* :ref:`developer_contract_optimizers`
* :ref:`developer_contract_transformers`

Any feedback or contributions will be appreciated.
`Project Repository <https://github.com/Dilosch03/HeteroSymNN>`_