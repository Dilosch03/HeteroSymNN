.. _custom_training_loop:

Custom Training Loops
=====================

HeteroSymNN provides a highly modular and granular API that is exceptionally well-suited for writing custom training loops. While the high-level :customref:`Wrapper <wrappers>` agent makes simple training effortless, advanced use-cases (such as custom learning rate scheduling, gradient clipping, or complex multi-loss aggregation) require manual loop control.

The :customref:`BaseNetwork <base-network>` architecture mirrors modern deep learning frameworks by decoupling the forward pass, backward pass, loss computation, and parameter updates.

1. The Granular API
-------------------

The framework exposes all necessary low-level primitives for a custom training loop:

* **Hardware Abstraction**: :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.cast_arrays` and :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.asnumpy` ensure seamless transitions between CPU (NumPy) and GPU (CuPy) backends without changing loop logic.
* **Forward Pass**: The :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.forward` method passes data layer-by-layer without implicit loss calculation.
* **Loss Function Decoupling**: Loss computation and error backpropagation are explicitly handled by the loss object (e.g., :meth:`~HeteroSymNN.Core.losses.Loss.forward` and :meth:`~HeteroSymNN.Core.losses.Loss.backward`).
* **Backward Pass**: :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.backward` propagates error derivatives backward through the network.
* **Parameter Updates**: :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.update_params` relies on the internal optimizer state to update network weights and biases.

2. Important Considerations
---------------------------

When writing a custom loop, keep the following internal mechanics in mind:

* **State Tracking**: :attr:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.num_completed_train_iterations` and :attr:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.num_completed_epochs` are manually incremented in the built-in :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.train_step` and :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.train` methods. If you bypass these methods, you must update these counters yourself for accurate metadata tracking.
* **Hardware Device Syncing**: The built-in methods frequently enforce device syncing. If switching hardware states midway, ensure your tensors are in the device that is going to be use for computation using the :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.to` method with the ``device`` string as argument. 
* **Gradient Access**: After :meth:`~HeteroSymNN.Core.Nets.base_classes.BaseNetwork.backward` is called, the gradient fed into the entire network is stored internally. If you need to access or modify weight gradients *before* the optimizer step, you must access them through the individual layers.

3. Example: Writing a Custom Loop
---------------------------------

Below is a safe and standard structure for a custom training loop leveraging the low-level primitives:

.. code-block:: python

    from HeteroSymNN.Core.Nets import MLP

    # 1. Initialize a model
    network = MLP(nodes_structure=[10, 25, 1])

    epochs = 100
    
    # Move to CPU or GPU depending on the computational method
    network.to("device")

    # Assuming custom_dataloader yields batches of x (inputs) and y (targets)
    for epoch in range(epochs):
        for x_batch, y_batch in custom_dataloader:
            # 1. Hardware abstraction: Send data to current backend (CPU/GPU)
            x, y = network.cast_arrays(x_batch, y_batch)
            
            # 2. Forward pass
            y_pred = network.forward(x)
            
            # 3. Loss Calculation
            # Compute loss value (for logging/tracking)
            loss_val = network.loss_function.forward(y_pred, y)
            
            # Compute the error to backpropagate
            error = network.loss_function.backward(y_pred, y)
            
            # (Optional) Mid-loop tampering could happen here, e.g., combining losses
            
            # 4. Backward Pass
            network.backward(error)
            
            # (Optional) Gradient clipping or modifications
            
            # 5. Parameter Update
            network.update_params()
            
            # Manually track training steps
            network.num_completed_train_iterations += 1

        #Manual tracking of epochs
        network.num_completed_epochs += 1
