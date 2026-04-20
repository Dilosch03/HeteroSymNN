.. _data_transformers:

Data Transformers
=================

Because HeteroSymNN allows for unbounded, custom symbolic activation functions (like ``x^3`` or ``exp(x)``), raw input data can easily cause numerical instability or exploding gradients during the JIT-compiled backward pass. The utilities module provides standardized data transformers to ensure inputs and outputs remain within safe, mathematically stable bounds.

.. note::
   **Extending the Framework:** Need to build a custom data scaler? Jump straight to the :ref:`developer_contract_transformers`.

.. currentmodule:: HeteroSymNN.API.data_transformers

Data Transformer Base Contract
------------------------------

.. autoclass:: DataTransformer
   :members:
   :undoc-members:

   The abstract foundation for all data scaling operations. When using the :class:`~HeteroSymNN.API.wrappers.Wrapper` API, transformers are invoked automatically before the forward pass (to scale inputs) and after predictions (to descale outputs). All standard transformers inherit from this class.

Standard Transformers
---------------------

The following transformers scale data automatically during the ``fit`` phase and handle descale operations during predictions.

.. automodule:: HeteroSymNN.API.data_transformers
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: DataTransformer


.. _developer_contract_transformers:

Developer Contract: Subclassing Transformers
--------------------------------------------

If you need a highly specific scaling distribution, you can build custom scaling logic by subclassing ``DataTransformer``. You must adhere to the following two lifecycles to integrate safely with the framework's state machine.

.. admonition:: Phase 1: The Execution Cycle (Math & State)
   :class: note

   Transformers are inherently stateful. They must lock their mathematical parameters after analyzing the training data.

   * **``fit(data)``**: Extract your statistical constants (e.g., dataset minimums, maximums, or variances). **Crucial:** You must call ``super().fit(data)`` at the end of this method to toggle the internal ``_fitted`` state flag.
   * **``transform(data)``** & **``inverse_transform(data)``**: Apply and reverse the scaling logic. You must call the ``super()`` equivalents first to trigger the runtime guardrails (which raise a :exc:`~HeteroSymNN.exceptions.RuntimeStateError` if transformation is attempted before fitting).

.. admonition:: Phase 2: Serialization & Parameter Loading
   :class: note

   Because transformers hold *learned* states (like the dataset's global maximum), those states must be saved alongside the network's weights.

   * **``get_config(self)``**: Must call ``super().get_config()`` and then inject your learned statistical constants into the returned dictionary (e.g., ``{"min": self._min}``).
   * **``set_config(self, config)``**: Must unpack the dictionary and overwrite the internal state variables so the exact same scaling can be applied to new data after the model is loaded from disk.