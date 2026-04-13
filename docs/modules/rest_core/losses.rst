.. _losses:

Loss Functions
==============

In HeteroSymNN, loss functions are parsed and compiled symbolically. This ensures that the exact symbolic derivative of your chosen loss function is calculated and fused directly into the backward pass, guaranteeing mathematical precision without the overhead of runtime autograd tracking.

.. note::
   **Extending the Framework:** Are you looking to build your own custom loss function or integrate a new mathematical graph? Jump straight to the :ref:`developer_contract_losses`.

.. currentmodule:: HeteroSymNN.Core.losses

Core Architecture (Base Classes)
--------------------------------

The following are the foundational objects powering the mathematical evaluations. All standard losses inherit from these blueprints.

.. autoclass:: Loss
   :members:
   :undoc-members:

.. autoclass:: FlexibleLoss
   :members:
   :undoc-members:
   :show-inheritance:


Standard Loss Functions
-----------------------

The following loss functions are pre-compiled and ready for use in standard regression or classification tasks. Because they inherit from the base classes above, only their unique properties are listed.

.. automodule:: HeteroSymNN.Core.losses
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Loss, FlexibleLoss

.. _developer_contract_losses:

Developer Contract: Extending Losses
---------------------------------------

HeteroSymNN provides two distinct pathways for building custom loss functions, depending on whether you want to leverage the JIT compiler or manually control the hardware routing.

.. admonition:: Pathway 1: Inheriting from "FlexibleLoss" (Recommended)
   :class: note

   For 99% of use cases, you should inherit from :class:`~HeteroSymNN.Core.losses.FlexibleLoss` You **do not** need to write custom forward or backward passes. You simply provide the mathematical formula as a string, and the :class:`~HeteroSymNN.JIT.compiler.SymbolicJITCompiler` classwill automatically derive the gradients.

   **The Execution Cycle:**
   You only need to invoke ``super().__init__()`` with your string formula and any required static constants.

   .. code-block:: python

      class CustomPenaltyLoss(FlexibleLoss):
          def __init__(self, penalty_weight: float = 1.0, computational_method=None, gpu_id=0):
              # Define the mathematical logic using standard SymPy syntax.
              formula = "(y_pred - y_true)**2 + (penalty * Abs(y_pred))"
              
              # Pass the constants directly to the JIT compiler
              super().__init__(
                  formula=formula, 
                  constants={"penalty": penalty_weight},
                  computational_method=computational_method,
                  gpu_id=gpu_id
              )

.. admonition:: Pathway 2: Inheriting from "Loss" (Advanced Hardware Control)
   :class: warning

   If you are bypassing the symbolic JIT compiler entirely (e.g., implementing a highly specific non-differentiable routing mechanism), you must inherit directly from the :class:`~HeteroSymNN.Core.losses.Loss` base class. 

   **The Hardware & Execution Lifecycle:**
   Because you are bypassing the automatic compiler, your subclass must manually implement hardware state synchronization to prevent memory crashes during device switching.

   * **``forward()`` & ``backward()``**: Must execute the math and return exactly shaped :data:`~HeteroSymNN.types.BackendArray` objects.
   * **``_change_COMPUTATIONAL_METHOD(new_method, gpu_id)``**: You must intercept this call to migrate any of your internal loss constants from Host RAM (NumPy) to Device VRAM (CuPy).
   * **``set_gpu_id(new_id)``**: Must update the pointer for the active hardware device.