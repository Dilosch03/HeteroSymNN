.. _losses:

Loss Functions
==============

The Concept
-----------
In HeteroSymNN, loss functions are parsed and compiled symbolically. This ensures that the exact symbolic derivative of your chosen loss function is calculated and fused directly into the backward pass, guaranteeing absolute mathematical precision without the overhead of runtime autograd tracking.

Standard Loss Functions
-----------------------
The following loss functions are pre-compiled and ready for use. Because they inherit from the base classes, only their unique equations and properties are listed.

.. automodule:: HeteroSymNN.Core.losses
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Loss, FlexibleLoss

Custom Loss Functions (The Easy Way)
------------------------------------
You **do not** need to write custom Python classes to experiment with new loss functions. 

Because of the symbolic JIT compiler, you can simply instantiate the :class:`~HeteroSymNN.Core.losses.FlexibleLoss` class directly, pass it any valid SymPy mathematical string, and the engine will automatically handle the compilation and derivatives for you.

.. note::
   But if you want a reusable loss function or implement a highly complex objective function check the :ref:`developer_contract_losses` section.


.. code-block:: python

    from HeteroSymNN.Core.losses import FlexibleLoss
    from HeteroSymNN.API import Wrapper

    # 1. Define your custom math and constants instantly
    my_custom_loss = FlexibleLoss(
        formula="(y_pred - y_true)**2 + (penalty * Abs(y_pred))",
        constants={"penalty": 1.5}
    )

    # 2. Pass it directly to your wrapper
    agent = Wrapper(model, work_type="reg", loss_function=my_custom_loss)


API Reference (Base Classes)
----------------------------
.. autoclass:: HeteroSymNN.Core.losses.Loss
   :members:
   :undoc-members:

.. autoclass:: HeteroSymNN.Core.losses.FlexibleLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. _developer_contract_losses:

Developer Contract: Extending Losses
------------------------------------

If you want to package a custom loss function into a **permanent, reusable class** (e.g., to share in a library), or if you are building a highly complex objective function that requires manual hardware routing, you should follow one of these two developer pathways.

.. admonition:: Pathway 1: Inheriting from "FlexibleLoss" (Permanent Symbolic Classes)
   :class: note

   This is the recommended approach for creating reusable loss packages. By inheriting from :class:`~HeteroSymNN.Core.losses.FlexibleLoss`, you hardcode the formula but allow users to pass the dynamic constants, while the framework still handles all C++/CUDA compilation automatically.

   .. code-block:: python

      from HeteroSymNN.Core.losses import FlexibleLoss

      class CustomPenaltyLoss(FlexibleLoss):
          def __init__(self, penalty_weight: float = 1.0, computational_method=None, gpu_id=0):
              
              # 1. Define the permanent mathematical logic
              formula = "(y_pred - y_true)**2 + (penalty * Abs(y_pred))"
              
              # 2. Pass the user's dynamic constants to the JIT compiler
              super().__init__(
                  formula=formula, 
                  constants={"penalty": penalty_weight},
                  computational_method=computational_method,
                  gpu_id=gpu_id
              )

.. admonition:: Pathway 2: Inheriting from "Loss" (Advanced Hardware Control)
   :class: warning

   If you are bypassing the symbolic JIT compiler entirely (e.g., implementing a highly specific, non-differentiable routing mechanism), you must inherit directly from the :class:`~HeteroSymNN.Core.losses.Loss` base class. 

   Because you are bypassing the automatic compiler, your subclass must manually implement hardware state synchronization:

   * **Initialization**: If you override ``__init__``, you **must** call ``super().__init__(computational_method, gpu_id)``.
   * **``forward(y_pred, y_true)`` & ``backward(y_pred, y_true)``**: Must execute the math and return exactly shaped :type:`~HeteroSymNN.types.BackendArray` objects. Raw NumPy arrays will crash the GPU pipeline.
   * **``_change_COMPUTATIONAL_METHOD(new_method, gpu_id)``**: You must intercept this call to migrate any of your internal loss constants from Host RAM (NumPy) to Device VRAM (CuPy).
   * **``set_gpu_id(new_id)``**: Must update the internal pointer for the active hardware device.