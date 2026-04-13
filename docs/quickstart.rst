.. _quickstart:

Quickstart
==========

HeteroSymNN utilizes a declarative architecture. You define the *structure* and the *mathematical symbols*, and the JIT compiler handles the underlying memory and execution logistics. The high-level API is designed to mirror Scikit-Learn for ease of use.

The "Cocktail" Layer Concept
----------------------------

In traditional neural networks, an entire layer applies a single activation function (e.g., ReLU). In HeteroSymNN, you can construct "Cocktail Layers"—layers composed of distinctly different mathematical functions operating side-by-side.

Basic Training Example
----------------------

This example demonstrates how to build a network predicting continuous values (Regression) using a mix of periodic and parameterized activation functions.

.. code-block:: python

    from HeteroSymNN.API.wrappers import Wrapper
    form HeteroSymNN.Core.Nets import Dense

    # 1. Define the Architecture & Symbolic Activations
    # nodes_structure: Defines the input, hidden, and output sizes.
    # activation_config: Maps to the layers (excluding the input layer).
    model = Dense(
        nodes_structure=[10, 25, 25, 1],
        activation_config=[
            "sin(num)",                             # Layer 1: Pure Sine wave
            "num",                                  # Layer 2: Linear pass-through
            ("tanh(num)*a", {"a": 2.0})             # Layer 3: Parameterized hyperbolic tangent
        ],
        training_mode="mini-batch",
        batch_size=32,
        num_training_iter=200
    )

    # 2. Wrap the model for automated memory management and training
    # "reg" indicates Regression. Use "class" for Classification.
    agent = Wrapper(model, work_type="reg")

    # 3. Fit the model to the data
    # The Wrapper automatically handles data formatting and scaling
    # X_train shape: (samples, 10), y_train shape: (samples, 1)
    loss_history = agent.fit(X_train, y_train)

    # 4. Generate Predictions
    predictions = agent.predict(X_test)

    # 5. Evaluate Metrics (Calculates R2, MSE, MAE, etc.)
    metrics = agent.regression_test_accuracy(X_test, y_test)
    print(metrics)

Understanding the Configuration
-------------------------------

Notice the ``activation_config`` list in the example above:

* Strings like ``"sin(num)"`` tell the symbolic compiler to parse a standard mathematical function, utilizing ``num`` as the standard variable for the incoming node value.
* Tuples like ``("tanh(num)*a", {"a": 2.0})`` allow you to define custom mathematical expressions while explicitly passing static constants (like ``a``) into the JIT-compiled C++ or CUDA kernel.