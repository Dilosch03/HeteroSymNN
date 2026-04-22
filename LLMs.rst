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

- **STRICT DEPENDENCY RULE**: NEVER import `torch`, `tensorflow`, or `keras`.
- **Used Backends**: The only backends used in this project are `numpy`, `cupy`, and `sympy`.
- **Mathematical Representation**: Mathematical activations and custom losses are passed as standard SymPy strings. Ensure you provide valid SymPy syntax.
- **Hardware Fallback**: The framework handles routing automatically (`GPU_CUDA -> CPU_JIT -> CPU_PYTHON`). Do not write manual device-transfer logic or force `.get()`/`.set()` calls unless explicitly required to bypass the agnostic `BackendArray`.
    **Note**: Currenlty "CPU_JIT" has been desabled due to poor performance. for CPU computation use "CPU_PYTHON" instead.

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
    from HeteroSymNN.API.wrappers import Wrapper
    from HeteroSymNN.API.data_transformers import MinMaxScaler

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
- ``LayerValues``: ``tuple[list[float], list[list[float]], list[list[float]]]`` - Ordered parameters: ``(Biases, Weights, ConnectionMask)``. will be depracted in future versions.   
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
*Note*: ``Initializer`` and ``BaseInitializer`` are strictly for subclassing, not direct instantiation.
- **Methods**: ``generate_binary_mask(shape: list[int]) -> np.ndarray``, ``generate_constant(shape: list[int], value: float = 0.0) -> np.ndarray``, ``generate_from_distribution(shape: list[int], fan_in: int, fan_out: int) -> np.ndarray``, ``get_config() -> dict[str, Any]``

*Subclass Note*: Standard initializers (HeNormal, HeUniform, LecunNormal, XavierNormal, XavierUniform) accept ``connection_density``. ``Orthogonal`` accepts ``gain``; ``RandomNormal`` accepts ``mean``/``stddev``; ``RandomUniform`` accepts ``minval``/``maxval``.
*Subclasses*: ``RandomNormal``, ``RandomUniform``, ``XavierUniform``, ``XavierNormal``, ``HeUniform``, ``HeNormal``, ``LecunNormal``, ``Orthogonal``.

3. API Wrapper
~~~~~~~~~~~~~~
**Instancing**: ``wrappers.Wrapper(model: BaseNetwork,work_type:Literal["class","reg"],input_transformer: Optional[data_transformers.DataTransformer] = None, output_transformer: Optional[data_transformers.DataTransformer] = None)``
- **Methods**: ``fit(X: list, y: list, epochs: int = None, ...) -> list[float]``, ``predict(data: list) -> np.ndarray``, ``test_accuracy(X: list, y: list) -> Union[dict, tuple]``, ``load_training(X: np.ndarray, y: np.ndarray) -> None``, ``run_training(num_iterations: int, batch_size: int = None) -> None``, ``classification_test_accuracy(X: np.ndarray, y: np.ndarray) -> tuple[dict, Any]``, ``regression_test_accuracy(X: np.ndarray, y: np.ndarray) -> dict``, ``save_model(path: str) -> None``, ``load_model(path: str) -> None``

4. Data Transformers
~~~~~~~~~~~~~~~~~~~~
**Base Class**: ``utilities.DataTransformer()``
*Note*: This is strictly for subclassing, not direct instantiation.
- **Methods**: ``fit(data: np.ndarray) -> None``, ``transform(data: np.ndarray) -> np.ndarray``, ``inverse_transform(data: np.ndarray) -> np.ndarray``, ``get_config() -> dict[str, float]``, ``set_config(config: dict[str, float]) -> None``
*Subclasses*: ``MinMaxScaler``

5. Optimizers
~~~~~~~~~~~~~
**Base Class**: ``optimizers.Optimizer(learning_rate: float = None, computational_device: Literal["GPU", "CPU"] = None, device_id: int = None)``
*Note*: This is strictly for subclassing, not direct instantiation.
- **Methods**: ``get_config() -> dict[str, Any]``
*Subclasses*: ``AdamOptimizer``, ``SgdOptimizer``.

6. Losses
~~~~~~~~~
**Base Class**: ``losses.Loss()``
*Note*: This is strictly for subclassing, not direct instantiation.
- **Methods**: ``forward(y_pred: BackendArray, y_true: BackendArray) -> BackendArray``, ``backward(y_pred: BackendArray, y_true: BackendArray) -> BackendArray``, ``set_gpu_id(new_id: int) -> None``, ``get_config() -> dict[str, Any]``

**FlexibleLoss**: Main class for custom losses.
``losses.FlexibleLoss(loss_expression: str, constants: dict[str, float] = None, computational_method: Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None, gpu_id: int = 0)``
*Subclasses*: ``MSELoss``, ``MAELoss``, ``HuberLoss``, ``BinaryCrossEntropy``.

7. Layers
~~~~~~~~~
**Base Class**: ``layers.BaseLayer(num_inputs: int, layer_configuration: LayerConstruction, batch_size: int = 1, gpu_id: int = 0)``
*Note*: This is strictly for subclassing, not direct instantiation.
- **Methods**: ``forward(input_values: BackendArray) -> BackendArray``, ``backward(error_values: BackendArray) -> BackendArray``, ``set_parameters(params: dict[str, np.ndarray]) -> None``, ``get_parameters() -> dict[str, np.ndarray]``, ``set_connection_mask(connection_mask: np.ndarray) -> None``, ``get_config() -> dict[str, Any]``
- **Forbidden**: Direct attribute mutation of layer weights/biases. Always use ``set_parameters(params: dict)`` to ensure cross-device consistency.
*Subclasses*: ``LinearLayer``.

Settings & Configuration
------------------------
The global ``settings`` object (imported from ``HeteroSymNN.config``) controls framework-level behavior.

**Key Properties**:

- ``use_kernel_cache``: ``bool`` - Whether to cache created kernels.
- ``n_jobs``: ``int`` - Number of CPU threads allowed.
- ``warning_level``: ``str`` - Strictness of warnings (``"ignore"``, ``"warn"``, ``"error"``).
- ``default_compute_method``: ``str`` - The default computational method (``"GPU_CUDA"``, ``"CPU_JIT"``, ``"CPU_PYTHON"``).
- ``cpu_cache_dir``: ``Path`` - Location for storing JIT-compiled CPU kernels.

**Key Methods**:

- ``set_warning_level(value: Literal["ignore","warn","error"])``
- ``set_default_compute_method(method: Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"])``
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

Example Framework Flow
----------------------
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