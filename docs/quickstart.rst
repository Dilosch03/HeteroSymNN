.. _quickstart:

Quickstart
==========

HeteroSymNN utilizes a declarative architecture designed for simplicity and extreme flexibility. You define the structure and the mathematical symbols, and the JIT compiler handles the underlying memory, gradients, and execution logistics.

The API provides different levels of abstraction depending on how much granular control you need over the mathematical laws of your network.

1. The Standard MLP (Homogeneous Network)
------------------------------------------

If you just need a standard, high-performance neural network where all hidden layers share the same activation function, use the MLP (Multi-Layer Perceptron) class. This is the simplest builder and mirrors traditional deep learning frameworks.

.. code-block:: python

    from HeteroSymNN.Core.Nets import MLP
    from HeteroSymNN.API import Wrapper

    # Define a network with 10 inputs, two hidden layers of 25 nodes, and 1 output.
    # By default, all hidden layers will use "relu" and the output uses "num" (linear).
    model = MLP(
        nodes_structure=[10, 25, 25, 1],
        activation="relu"
    )

    # Wrap the model for Scikit-Learn style training
    agent = Wrapper(model, work_type="reg") # "reg" for Regression

    # Train instantly
    agent.fit(X_train, y_train, epochs=100)

2. Layer-Level Heterogeneity (The "Cocktail" Network)
------------------------------------------------------

In standard frameworks, mixing different activation functions usually requires custom boilerplate classes. With HeteroSymNN's Dense builder, you can effortlessly assign different mathematical strings to different layers.

Here is how to create a network that uses periodic functions (Sine) in the first hidden layer, and parameterized functions in the second:

.. code-block:: python

    from HeteroSymNN.Core.Nets import Dense
    from HeteroSymNN.API import Wrapper

    # Define the network topology and layer-wise activations
    model = Dense(
        nodes_structure=[10, 25, 25, 1],
        activation_config=[
            "sin(num)",                        # Hidden Layer 1: Sine wave
            "Max(0, num)",                     # Hidden Layer 2: Standard ReLU
            ("tanh(num) * a", {"a": 2.0})      # Output Layer: Parameterized Tanh
        ]
    )

    agent = Wrapper(model, work_type="reg")
    agent.fit(X_train, y_train, epochs=100)

3. Node-Level Heterogeneity (Deep Customization)
-------------------------------------------------

The true superpower of the HeteroSymNN JIT Compiler is that it is not restricted to layers. You can define a different mathematical activation function for every single neuron in a layer. The engine fuses all of these distinct instructions into a single C++/CUDA kernel launch, avoiding execution penalties.

To achieve this granular control, use the node-level configuration arrays:

.. code-block:: python

    from HeteroSymNN.Core.Nets import HeteroDense
    # Define a small network: 2 inputs, a hidden layer with 3 neurons, 1 output
    # We explicitly define the math for each of the 3 hidden neurons:
    hidden_activations = [
        ("sin(num)",{}),                        # Neuron type 1: Periodic sine wave
        ("Max(0, num)", {}),                     # Neuron type 2: Standard ReLU
        ("exp(num * beta)", {"beta": -0.5})    # Neuron type 3: Parameterized Exponential
    ]*4

    # The output layer (1 neuron) uses a standard linear activation
    output_activations = [("num", {})]

    # Construct the fully heterogeneous network
    hetero_model = HeteroDense(
        nodes_structure=[2, 12, 1],
        detailed_activations=[hidden_activations, output_activations]
    )

4. Zero-Recompile Tuning (Dynamic Constants)
--------------------------------------------

Notice the variables ``a`` and ``beta`` in the examples above. You are not forced to hard-code numerical constraints in your strings.

HeteroSymNN treats these symbolic constants as mutable kernel arguments. This means you can update these hyperparameters dynamically on the fly without triggering a slow C++/CUDA recompilation.

.. code-block:: python

    # Update the 'beta' constant for a specific layer/neuron in real-time
    # The framework routes this directly to the compiled C++ kernel memory.

    # Example: Update the 'beta' parameter we defined in the HeteroDense model
    # Format: {Layer_Index: list or single of (Node_Index, Constant_Name, New_Value))}
    hetero_model.change_constants({0: (2, "beta", -0.9)})

This feature is exceptionally powerful for hyperparameter grid searching or Evolutionary Algorithms, where constants must mutate thousands of times per second.

5. Saving & Loading Custom Models (The Registry)
------------------------------------------------

HeteroSymNN allows you to easily serialize your wrapped models to disk. 

However, if you extended the framework by building a **custom class** (e.g., a custom Loss function or custom Layer) and used it in your model, you **must register the class pointer** before loading the model back from disk. This teaches the deserializer how to rebuild your custom architecture.

.. code-block:: python

    from HeteroSymNN.API import Wrapper, registry
    from my_custom_code import MyCustomLoss

    # 1. Register your custom class pointer (NOT an instance!) BEFORE loading
    registry.add_loss_func(MyCustomLoss)

    # 2. Safely load the model that was trained with MyCustomLoss
    loaded_agent = Wrapper.load_model("my_custom_model.symnn")

The `:class:~HeteroSymNN.API.registry` provides methods for all extendable components: `add_net()`, `add_layer()`, `add_loss_func()`, `add_optimizer()`, `add_initializer()`, and `add_data_transformer()`.