HeteroSymNN v0.3.0 - AI Developer Guide
=======================================

Welcome, AI Agent! This document serves as your mini-documentation and context guide for working within the **HeteroSymNN** repository. 

**Identity**: HeteroSymNN is a JIT-compiled, Differentiable Compiler and Neuro-Symbolic engine. It relies on dynamically compiled mathematical expressions rather than static computational graphs.

What You Can Do
---------------
As an LLM contributing to or using HeteroSymNN, you have the capability to:

1. **Design Custom Architectures**: Build homogeneous networks (`Dense`, `MLP`) or highly heterogeneous networks (`HeteroDense`) where every node can have a different activation function.
2. **Define Symbolic Math**: Implement novel activation functions, loss functions, and node operations simply by passing SymPy-compatible string formulas (e.g., `"sin(x) * alpha"`).
3. **Train Models**: Utilize the `API.wrappers.Wrapper` for automated data handling and training, or write custom training loops using core networks and optimizers.
4. **Develop Core Features**: Implement new `Initializers`, `Optimizers`, or `Loss` functions by adhering to the base class architectures in `HeteroSymNN.Core`.
5. **Zero-Recompile Tuning**: Update constants within symbolic expressions on-the-fly without recompiling the underlying C++/CUDA kernels (using `set_constants`).

How to Do It (System Directives)
--------------------------------
You must adhere to the following rules when writing or modifying code:

- **Mathematical Representation**: Mathematical activations and custom losses are passed as standard SymPy strings. Ensure you provide valid SymPy syntax.

Anti-Patterns & Forbidden Actions
---------------------------------
To ensure compatibility, cross-device consistency, and hardware safety, **NEVER** do the following:

- **NEVER import External DL Frameworks**: Do not import ``torch``, ``tensorflow``, or ``keras``. The only backends used in this project are ``numpy``, ``cupy``, and ``sympy``.
- **NEVER write manual device-transfer logic**: Do not force ``.get()`` / ``.set()`` calls unless explicitly required to bypass the agnostic ``BackendArray``. The framework handles routing automatically (``GPU_CUDA -> CPU_PYTHON``).
- **NEVER mutate layer weights/biases directly**: Direct attribute mutation of layer parameters (e.g., ``layer.weights = new_weights``) is forbidden. Always use ``set_parameters(params: dict)`` and ``get_parameters()`` to ensure cross-device consistency.
- **NEVER directly instantiate Abstract Base Classes**: The following base classes are strictly for subclassing and must not be directly instantiated:
  - ``initializers.Initializer`` & ``initializers.BaseInitializer``
  - ``utilities.DataTransformer``
  - ``optimizers.Optimizer``
  - ``losses.Loss``
  - ``layers.BaseLayer``

Directory Structure Guide
-------------------------
.. code-block:: text

    /HeteroSymNN
    ├── __init__.py
    ├── types.py             # Type aliases
    ├── config.py            # (Optional) Global settings 
    ├── exceptions.py        # Used exceptions like CompilationError, ShapeMismatchError, BackendNotAvailableError, etc.
    │
    ├── /Core
    │   ├── __init__.py
    │   ├── layers.py        # <--- The 'Layer' class (The Atom)
    │   ├── losses.py
    │   ├── initializers.py
    │   ├── optimizers.py
    │   │   
    │   └── /Nets            # <--- The Networks (The Molecules)
    │       ├── __init__.py
    │       ├── base.py      # Base 'NeuralNetwork' class (The Engine)
    │       ├── dense.py     # 'HeteroDense', 'Dense', 'MLP'
    │       ├── evo.py       # 'EvolvableDense'
    │       └── functional.py # 'SymbolicDense'
    │
    ├── /JIT
    │   ├── __init__.py
    │   ├── compiler.py      # The Director (manages the compilation process)
    │   └── codegen.py       # Holds the templates for the code generation for each backend.
    │
    ├── /Backends
    │   ├── __init__.py
    │   └── hardware.py      # Device selection logic
    │
    └── /API
        ├── __init__.py
        ├── registries.py
        ├── wrapper.py       # The High-Level 'Wraper' (Model Interface)
        └── data_transformers.py     # (New) Scalers, metric calculators, etc.

Import Structure
----------------
Always use the following paths for your imports:

.. code-block:: python

    # Core Networks
    from HeteroSymNN.Core.Nets import HeteroDense, Dense, MLP, BaseNetwork

    # API & High-Level Wrappers
    from HeteroSymNN.API import Wrapper
    from HeteroSymNN.API import MinMaxScaler

    # Subcomponents
    from HeteroSymNN.Core import losses
    from HeteroSymNN.Core import optimizers
    from HeteroSymNN.Core import initializers
    from HeteroSymNN.Core import layers

    # Type Hinting & Configurations
    from HeteroSymNN.types import FlexibleNodeConfig, NodeConfig, BackendArray, ConstantToUpdate, LayerValues, LayerConstruction
    from HeteroSymNN.config import settings

Type Definitions
----------------
Use these types for type hinting and understanding expected inputs:

- ``NodeConfig``: ``tuple[str, dict[str, float]]`` - Defines a node's math string and its constants.
- ``FlexibleNodeConfig``: ``Union[str, NodeConfig]`` - Can be a simple string (e.g., ``"relu"``) or a full ``NodeConfig``.
- ``LayerValues``: ``tuple[list[float], list[list[float]], list[list[float]]]`` - Ordered parameters: ``(Biases, Weights, ConnectionMask)``.
- ``LayerConstruction``: ``tuple[list[NodeConfig], Initializer]`` - Blueprint for constructing a layer.
- ``BackendArray``: ``np.ndarray`` - Agnostic array type (resolves to NumPy or CuPy at runtime).
- ``ConstantToUpdate``: ``tuple[int, str, float]`` - Used for zero-recompile tuning: ``(Node Index, Constant Name, New Value)``.

Common Symbolic Formulas
------------------------
HeteroSymNN internally defines common activation and loss formulas that you can reference by their keys. The JIT compiler parses these directly into C++/CUDA code.

**Activations**:

- ``"relu"``: ``Max(0.0, num)``
- ``"sigmoid"``: ``1 / (1 + exp(-num))``
- ``"swish"`` / ``"SiLU"``: ``num / (1 + exp(-num))``
- ``"softplus"``: ``log(1 + exp(num))``
- ``"mish"``: ``num * tanh(log(1 + exp(num)))``
- ``"gelu"``: ``0.5 * num * (1 + erf(num / sqrt(2.0)))``
- ``"linear"``: ``num``

**Losses**:

- ``"mse"``: ``(y_pred - y_true)**2``
- ``"mae"``: ``Abs(y_pred - y_true)``
- ``"huber"``: ``Piecewise((0.5 * (y_pred - y_true)**2, Abs(y_pred - y_true) <= 1.0), (Abs(y_pred - y_true) - 0.5, True))``
- ``"bce"``: ``-(y_true * log(y_pred + 1e-7) + (1 - y_true) * log(1 - y_pred + 1e-7))``

Component Structure & Instantiation
-----------------------------------
When interacting with or subclassing HeteroSymNN components, strictly utilize the following signatures to ensure hardware safety and correct execution order.

1. Networks
~~~~~~~~~~~
**Base Network**:
``Nets.BaseNetwork(network_structure: list[tuple[int, type[BaseLayer]]], extra_layer_parameters: list[dict[str, Any]], detailed_activations: list[list[NodeConfig]], initial_values: Optional[list[LayerValues]] = None, initializers: Optional[list[Initializer]] = None, learning_rate: float = 0.001, batch_size: int = 32, training_mode: Literal["batch", "mini-batch", "stochastic"] = "mini-batch", loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_epochs: int = 1000)``
- **Methods**: ``train(training_inputs: list, training_targets: list, ...) -> list[float]``, ``predict(input_values: list, to_cpu: bool = True) -> Union[np.ndarray, BackendArray]``, ``get_parameters() -> dict``, ``set_parameters(params: dict) -> None``, ``change_device(device: Literal["CPU", "GPU"]) -> None``, ``set_gpu_id(new_id: int) -> None``, ``change_constants(new_constants: dict) -> None``, ``get_config() -> dict[str, Any]``
- **Attributes (Read-Only)**: ``layers`` (Sequence[BaseLayer]), ``network_structure`` (Sequence[int]), ``gpu_id`` (int), ``batch_size`` (int), ``current_device`` (Literal["CPU", "GPU"]), ``computational_method`` (Literal["GPU_CUDA", "CPU_PYTHON"]), ``optimizer`` (Optimizer), ``loss_function`` (Loss), ``initializer`` (list[Initializer]), ``history_losses`` (list[float]), ``num_completed_train_iterations`` (int), ``num_completed_epochs`` (int)
- **Attributes (Read-Write)**: ``learning_rate`` (float), ``training_mode`` (Literal["batch", "mini-batch", "stochastic"]), ``num_training_epochs`` (int)

**HeteroDense**: 
``HeteroDense(nodes_structure: list[int], detailed_activations: list[list[NodeConfig]], initial_values: Optional[list[LayerValues]] = None, initializer: Optional[list[Initializer]] = None, learning_rate: float = 0.001, batch_size: int = 32, training_mode: str = "mini-batch", loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_training_iter: int = 1000)``

**Dense**: 
``Dense(nodes_structure: list[int], activation_config: list[FlexibleNodeConfig], initial_values: Optional[list[LayerValues]] = None, initializer: Optional[Union[Initializer, list[Initializer]]] = None, learning_rate: float = 0.001, training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic", batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_training_iter: int = 1000)``

**MLP**: 
``MLP(nodes_structure: list[int], activation: FlexibleNodeConfig = "relu", output_activation: FlexibleNodeConfig = "num", initializer: Optional[Initializer] = None, learning_rate: float = 0.001, training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic", batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_training_iter: int = 1000)``

*Important Note on Network Structure*: Technically, an "input layer" does not exist as a standalone component. The first element in ``nodes_structure`` strictly defines the number of input features. Therefore, lists like ``activation_config`` and ``detailed_activations`` must contain exactly **1 less element** than ``nodes_structure``.

2. Initializers
~~~~~~~~~~~~~~~
**Base Class**: ``initializers.Initializer(connection_density: float = 1.0)``
- **Methods**: ``generate_binary_mask(shape: list[int]) -> np.ndarray``, ``generate_constant(shape: list[int], value: float = 0.0) -> np.ndarray``, ``generate_from_distribution(shape: list[int], fan_in: int, fan_out: int) -> np.ndarray``, ``get_config() -> dict[str, Any]``
- **Attributes (Read-Only)**: ``connection_density`` (float), ``mean`` (float), ``stddev`` (float), ``min_val`` (float), ``max_val`` (float), ``gain`` (float)

*Subclass Note*: Standard initializers (HeNormal, HeUniform, LecunNormal, XavierNormal, XavierUniform) accept ``connection_density``. ``Orthogonal`` accepts ``gain``; ``RandomNormal`` accepts ``mean``/``stddev``; ``RandomUniform`` accepts ``minval``/``maxval``.
*Subclasses*: ``RandomNormal``, ``RandomUniform``, ``XavierUniform``, ``XavierNormal``, ``HeUniform``, ``HeNormal``, ``LecunNormal``, ``Orthogonal``.

3. API Wrapper
~~~~~~~~~~~~~~
**Instancing**: ``wrappers.Wrapper(model: BaseNetwork,work_type:Literal["class","reg"],input_transformer: Optional[data_transformers.DataTransformer] = None, output_transformer: Optional[data_transformers.DataTransformer] = None)``
- **Methods**: ``fit(X: list, y: list, epochs: int = None, ...) -> list[float]``, ``predict(data: list) -> np.ndarray``, ``test_accuracy(X: list, y: list) -> Union[dict, tuple]``, ``load_training(X: np.ndarray, y: np.ndarray) -> None``, ``run_training(num_iterations: int, batch_size: int = None) -> None``, ``classification_test_accuracy(X: np.ndarray, y: np.ndarray) -> tuple[dict, Any]``, ``regression_test_accuracy(X: np.ndarray, y: np.ndarray) -> dict``, ``save_model(path: str) -> None``, ``load_model(path: str) -> None``
- **Attributes (Read-Only)**: ``training_data`` (tuple[np.ndarray, np.ndarray]), ``training_data_norm`` (tuple[np.ndarray, np.ndarray])
- **Attributes (Read-Write)**: ``model`` (BaseNetwork), ``input_transformer`` (Optional[DataTransformer]), ``output_transformer`` (Optional[DataTransformer]), ``work_type`` (Literal["class", "reg"]), ``model_name`` (str)

4. Data Transformers
~~~~~~~~~~~~~~~~~~~~
**Base Class**: ``utilities.DataTransformer()``
- **Methods**: ``fit(data: np.ndarray) -> None``, ``transform(data: np.ndarray) -> np.ndarray``, ``inverse_transform(data: np.ndarray) -> np.ndarray``, ``get_config() -> dict[str, float]``, ``set_config(config: dict[str, float]) -> None``
- **Attributes (Read-Only)**: ``is_fitted`` (bool), ``min`` (float), ``max`` (float)
*Subclasses*: ``MinMaxScaler``

5. Optimizers
~~~~~~~~~~~~~
**Base Class**: ``optimizers.Optimizer(learning_rate: float = None, computational_device: Literal["GPU", "CPU"] = None, device_id: int = None)``
- **Methods**: ``get_config() -> dict[str, Any]``
- **Attributes (Read-Only)**: ``DEVICE_ID`` (int), ``CURRENT_DEVICE`` (Literal["CPU", "GPU"]), ``COMPUTATIONAL_DEVICE`` (Literal["GPU", "CPU"])
- **Attributes (Read-Write)**: ``learning_rate`` (float)
*Subclasses*: ``AdamOptimizer``, ``SgdOptimizer``.

6. Losses
~~~~~~~~~
**Base Class**: ``losses.Loss()``
- **Methods**: ``forward(y_pred: BackendArray, y_true: BackendArray) -> BackendArray``, ``backward(y_pred: BackendArray, y_true: BackendArray) -> BackendArray``, ``set_gpu_id(new_id: int) -> None``, ``get_config() -> dict[str, Any]``

**FlexibleLoss**: Main class for custom losses.
``losses.FlexibleLoss(loss_expression: str, constants: dict[str, float] = None, computational_method: Literal["GPU_CUDA","CPU_PYTHON"] = None, gpu_id: int = 0)``
- **Attributes (Read-Only)**: ``LOSS_EXPRESSION`` (str), ``COMPUTATIONAL_METHOD`` (Literal["GPU_CUDA","CPU_PYTHON"]), ``GPU_ID`` (int), ``COMPILER`` (SymbolicJITCompiler)
*Subclasses*: ``MSELoss``, ``MAELoss``, ``HuberLoss``, ``BinaryCrossEntropy``.

7. Layers
~~~~~~~~~
**Base Class**: ``layers.BaseLayer(num_inputs: int, layer_configuration: LayerConstruction, batch_size: int = 1, gpu_id: int = 0)``
- **Methods**: ``forward(input_values: BackendArray) -> BackendArray``, ``backward(error_values: BackendArray) -> BackendArray``, ``set_parameters(params: dict[str, np.ndarray]) -> None``, ``get_parameters() -> dict[str, np.ndarray]``, ``set_connection_mask(connection_mask: np.ndarray) -> None``, ``get_config() -> dict[str, Any]``
- **Attributes (Read-Only)**: ``initializer`` (Initializer), ``num_nodes`` (int), ``num_inputs`` (int), ``gpu_id`` (int), ``computational_method`` (Literal["GPU_CUDA", "CPU_PYTHON"]), ``activation_function_constants`` (BackendArray), ``current_device`` (Literal["CPU", "GPU"]), ``inicial_nodes_layer_configs`` (Sequence[NodeConfig])
*Subclasses*: ``LinearLayer``.

8. Registries
~~~~~~~~~~~~~
**Location**: ``HeteroSymNN/API/registries.py``
The ``registry`` object within this module stores class pointers essential for the serialization and deserialization of objects. Whenever you implement a new custom component (such as a Network, Layer, Initializer, Optimizer, Loss, or DataTransformer), you **must** use this ``registry`` object (e.g., ``registry.add_net(MyCustomNet)``) to register your custom classes. This ensures that model loading mechanisms can correctly identify and load your networks from saved ``.symnn`` files.
- **Methods**: ``add_initializer(custom_initializer: type[initializers.Initializer])``, ``add_loss_func(custom_loss: type[losses.Loss])``, ``add_optimizer(custom_optimizer: type[optimizers.Optimizer])``, ``add_layer(custom_layer: type[layers.BaseLayer])``, ``add_net(custom_net: type[Nets.BaseNetwork])``, ``add_data_transformer(custom_data_transformer: type[data_transformers.DataTransformer])``
- **Attributes (Read-Only)**: ``loss_fn_map`` (dict[str,type[losses.Loss]]), ``optimiers_map`` (dict[str,type[optimizers.Optimizer]]), ``initializers_map`` (dict[str,type[initializers.Initializer]]), ``layers_map`` (dict[str,type[layers.BaseLayer]]), ``net_map`` (dict[str,type[Nets.BaseNetwork]]), ``data_transformers_map`` (dict[str,type[data_transformers.DataTransformer]])

Extending Base Classes (Subclassing Requirements)
-------------------------------------------------
When implementing custom components by subclassing the framework's base classes, you must adhere to strict type preservation and initialization rules. A critical rule across all subclasses is the preservation of the ``BackendArray`` type. The framework relies on hardware-agnostic ``BackendArray`` objects (which resolve to either NumPy or CuPy arrays). A subclass that returns a plain ``np.ndarray`` from a method expecting a ``BackendArray`` might work on the CPU but will silently fail or cause data transfer bottlenecks on the GPU.

1. **Networks** (``BaseNetwork``)
- **Required Methods**: Override ``__init__`` to define the network structure.
- **super() Requirement**: **MUST** call ``super().__init__(network_structure, extra_layer_parameters, detailed_activations, ...)`` to initialize the architecture and memory allocation.

2. **Layers** (``BaseLayer``)
- **Required Methods**:
  - ``forward(self, input_values: BackendArray) -> BackendArray``
  - ``backward(self, error_values: BackendArray) -> BackendArray``
  - ``get_parameters(self) -> dict[str, np.ndarray]``
  - ``set_parameters(self, params: dict[str, np.ndarray]) -> None``
- **super() Requirement**: **MUST** call ``super().__init__(num_inputs, layer_configuration, batch_size, Gpu_id)``.
- **Type Preservation**: ``forward`` and ``backward`` **MUST** return ``BackendArray`` objects, preserving the backend type of their inputs.

3. **Loss Functions** (``Loss``)
- **Required Methods**:
  - ``forward(self, y_pred: BackendArray, y_true: BackendArray) -> BackendArray``
  - ``backward(self, y_pred: BackendArray, y_true: BackendArray) -> BackendArray``
- **super() Requirement**: **MUST** call ``super().__init__(computational_method, gpu_id)``.
- **Type Preservation**: Both methods **MUST** return ``BackendArray`` objects.

4. **Optimizers** (``Optimizer``)
- **Required Methods**:
  - ``_refresh_parameters(self, vector_format)``
  - ``_single_update(self, layer: BaseLayer, param_name: str, param: BackendArray, grad: BackendArray, mask: Union[float, BackendArray] = 1.0)``
- **super() Requirement**: **MUST** call ``super().__init__(learning_rate, computational_device, device_id)``.
- **Type Preservation**: In ``_single_update``, you **MUST** use the provided backend operations (via ``self.be``) or kernels to update ``param`` in-place or preserve its ``BackendArray`` type.

5. **Initializers** (``Initializer``)
- **Required Methods**:
  - ``generate_binary_mask(self, shape: list[int]) -> np.ndarray``
  - ``generate_constant(self, shape: list[int], value: float = 0.0) -> np.ndarray``
  - ``generate_from_distribution(self, shape: list[int], fan_in: int, fan_out: int) -> np.ndarray``
- **super() Requirement**: **MUST** call ``super().__init__()``.

6. **Data Transformers** (``DataTransformer``)
- **Required Methods**:
  - ``fit(self, data: np.ndarray) -> None``
  - ``transform(self, data: np.ndarray) -> np.ndarray``
  - ``inverse_transform(self, data: np.ndarray) -> np.ndarray``
  - ``get_config(self) -> dict[str, Any]``
  - ``set_config(self, config: dict[str, Any]) -> None``
- **super() Requirement**: **MUST** call ``super().__init__()``. Additionally, overriding methods like ``fit``, ``transform``, and ``inverse_transform`` should call their ``super()`` equivalents to correctly maintain the internal ``_fitted`` state.

Settings & Configuration
------------------------
The global ``settings`` object (imported from ``HeteroSymNN.config``) controls framework-level behavior.

**Key Properties**:

- ``use_kernel_cache``: ``bool`` - Whether to cache created kernels.
- ``n_jobs``: ``int`` - Number of CPU threads allowed.
- ``warning_level``: ``str`` - Strictness of warnings (``"ignore"``, ``"warn"``, ``"error"``).
- ``default_compute_method``: ``str`` - The default computational method (``"GPU_CUDA"``, ``"CPU_PYTHON"``).
- ``cpu_cache_dir``: ``Path`` - Location for storing JIT-compiled CPU kernels.

**Key Methods**:

- ``set_warning_level(value: Literal["ignore","warn","error"])``
- ``set_default_compute_method(method: Literal["GPU_CUDA","CPU_PYTHON"])``
- ``clear_kernel_cache(cache_type: Literal["ALL","CPU","GPU"] = "ALL")``
- ``set_cache_location(path: Union[str,Path], move_existing_cache: bool = False)``

Exception Handling
------------------
Wrap operations in the following custom exceptions from ``HeteroSymNN.exceptions``:

- ``HeteroSymNNError``: Base exception.
- ``BackendNotAvailableError`` / ``MethodMigrationError``: Hardware routing failures.
- ``CompilationWarning`` / ``JITError``: Errors when SymPy string fails to compile to C++/CUDA.
- ``ShapeMismatchError`` / ``LayerConfigurationError``: Network architecture dimension issues.
- ``WrapperError`` / ``LoadingError`` / ``SavingError``: API disk I/O issues.
- ``RuntimeStateError``: Execution methods called out of order (e.g., backward pass before forward pass).

Examples of the Framework Flow
-------------------------------
This illustrates the standard pipeline for initializing, wrapping, and training a model using the HeteroSymNN API:

.. code-block:: python

    import numpy as np
    from HeteroSymNN.Core.Nets import Dense
    from HeteroSymNN.API import Wrapper

    # 1. Define the network topology and custom symbolic activations
    model = Dense(
        nodes_structure=[10, 25, 25, 1],
        activation_config=[
            "sin(num)",                        # Hidden Layer 1: Sine wave
            "Max(0, num)",                     # Hidden Layer 2: Standard ReLU
            ("tanh(num) * a", {"a": 2.0})      # Output Layer: Parameterized Tanh
        ]
    )

    # 2. Wrap the model for higher-level API access
    # "reg" for Regression, "class" for Classification
    agent = Wrapper(model, work_type="reg") 

    # 3. Load data and train
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    
    agent.load_training(X_train, y_train)
    agent.run_training(num_iterations=100, batch_size=32)


.. code-block:: python

    from HeteroSymNN.Core.Nets.dense import HeteroDense
    # Define a small network: 2 inputs, a hidden layer with 3 neurons, 1 output
    # We explicitly define the math for each of the 3 hidden neurons:
    hidden_activations = [
        ["sin(num)",{}],                        # Neuron type 1: Periodic sine wave
        ["Max(0, num)",{}],                     # Neuron type 2: Standard ReLU
        ["exp(num * beta)", {"beta": -0.5}]    # Neuron type 3: Parameterized Exponential
    ]*4

    # The output layer (1 neuron) uses a standard linear activation
    output_activations = [["num",{}]]

    # Construct the fully heterogeneous network
    hetero_model = HeteroDense(
        nodes_structure=[2, 12, 1],
        detailed_activations=[hidden_activations, output_activations]
    )