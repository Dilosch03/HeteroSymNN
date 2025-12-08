Quickstart
==========

HeteroSymNN follows a Scikit-Learn style API. Here is how to create a "Cocktail Layer" that mixes periodic and linear features.

Basic Training Example
----------------------

.. code-block:: python

    from HeteroSymNN.API.wrapper import Wraper
    from HeteroSymNN.Core.Nets.neural_nets import FlexibleNN

    # 1. Define the architecture and activation functions
    # Note: You can mix strings like "sin(x)" with parameter tuples
    model = FlexibleNN(
        nodes_structure=[10, 25, 25, 1],
        activation_config=[
            "sin(x)",
            "num",
            ("tanh(z)*a", {"a": 2})
        ],
        training_mode="mini-batch",
        batch_size=32,
        num_treaning_iter=200
    )

    # 2. Wrap the model for training (Regression mode)
    agent = Wraper(model, work_type="reg")

    # 3. Load data and train
    # agent.load_training(X_train, y_train)
    # agent.run_training(num_iterations=100, batch_size=64)

How It Works
------------

HeteroSymNN acts as a **Differentiable Compiler**:

1. **Parse**: Accepts mathematical strings (e.g., ``"alpha * sin(x)"``) and parses them using SymPy.
2. **Derive**: Automatically calculates the symbolic derivative for backpropagation.
3. **Compile**: Generates C++ or CUDA code at runtime, creating a ``switch`` statement for distinct neuron instructions.
4. **Fuse**: Fuses memory access into a single kernel launch.