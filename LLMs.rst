HeteroSymNN v0.3.0 - LLM Context Documentation

System Directives for AI Code Generation

Identity: HeteroSymNN is a JIT-compiled, Differentiable Compiler and Neuro-Symbolic engine. It does not use static computational graphs.

STRICT DEPENDENCY RULE: NEVER import torch, tensorflow, or keras. HeteroSymNN relies strictly on numpy, cupy, and sympy.

Math Representation: Mathematical activations and custom losses are passed as standard SymPy strings (e.g., "sin(x) * alpha").

Hardware Fallback: The framework handles routing automatically (GPU_CUDA -> CPU_JIT -> CPU_PYTHON). Do not write manual device-transfer logic unless explicitly required.

1. Import Structure & Modules

Core Networks

from HeteroSymNN.Core.Nets.dense import HeteroDense, Dense, MLP

from HeteroSymNN.Core.Nets.base_classes import BaseNetwork

API & High-Level Wrappers

from HeteroSymNN.API.wrappers import Wrapper

from HeteroSymNN.API.utilities import DataTransformer

Subcomponents

from HeteroSymNN.Core import losses

from HeteroSymNN.Core import optimizers

from HeteroSymNN.Core import initializers

from HeteroSymNN.Core import layers

Type Hinting & Configurations

from HeteroSymNN.types import FlexibleNodeConfig, NodeConfig, BackendArray, ConstantToUpdate, LayerValues, LayerConstruction

from HeteroSymNN.config import settings

Type Definitions & Descriptions

NodeConfig: tuple[str, dict[str, float]] - Defines a node's math string and its constants.

FlexibleNodeConfig: Union[str, NodeConfig] - Can be a simple string (e.g., "relu") or a full NodeConfig tuple.

LayerValues: tuple[list[float], list[list[float]], list[list[float]]] - Ordered parameters: (Biases, Weights, ConnectionMask).

LayerConstruction: tuple[list[NodeConfig], Initializer] - Blueprint for constructing a layer.

BackendArray: np.ndarray - Agnostic array type (resolves to NumPy or CuPy at runtime).

ConstantToUpdate: tuple[int, str, float] - Used for zero-recompile tuning: (Node Index, Constant Name, New Value).

2. Component Structure & Instantiation

When interacting with or subclassing HeteroSymNN components, strictly utilize the following signatures to ensure hardware safety and correct execution order.

2.1 Initializers

General Instancing

initializers.Initializer(connection_density: float = 1.0)

General Attributes

None

General Methods

generate_binary_mask(shape: list[int]) -> np.ndarray

generate_constant(shape: list[int], value: float = 0.0) -> np.ndarray

generate_from_distribution(shape: list[int], fan_in: int, fan_out: int) -> np.ndarray

get_config() -> dict[str, Any]

Subclass Instantiation Differences

Orthogonal: initializers.Orthogonal(connection_density: float = 1.0, gain: float = 1.0)

RandomNormal: initializers.RandomNormal(connection_density: float = 1.0, mean: float = 0.0, stddev: float = 0.05)

RandomUniform: initializers.RandomUniform(connection_density: float = 1.0, minval: float = -0.05, maxval: float = 0.05)

(Standard initializers like HeNormal and GlorotUniform typically only accept connection_density)

2.2 Networks

General Instancing

Nets.BaseNetwork(network_structure: list[tuple[int, type[BaseLayer]]], extra_layer_parameters: list[dict[str, Any]], detailed_activations: list[list[NodeConfig]], initial_values: Optional[list[LayerValues]] = None, initializers: Optional[list[Initializer]] = None, learning_rate: float = 0.001, batch_size: int = 32, training_mode: Literal["batch", "mini-batch", "stochastic"] = "mini-batch", loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_epochs: int = 1000)

General Attributes & Properties

training_mode: Literal

$$"batch", "mini-batch", "stochastic"$$

General Methods

forward(X: BackendArray) -> BackendArray

backward(error: BackendArray) -> None

change_device(new_device: Literal["CPU", "GPU"], gpu_id: int) -> None

set_constants(updates: list[ConstantToUpdate]) -> None

get_config() -> dict[str, Any]

Subclass Instantiation Differences

HeteroDense: HeteroDense(nodes_structure: list[int], detailed_activations: list[list[FlexibleNodeConfig]], initial_values: Optional[list[LayerValues]] = None, initializer: Optional[list[Initializer]] = None, learning_rate: float = 0.001, training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic", batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_epochs: int = 1000)

Dense: Dense(nodes_structure: list[int], activation: FlexibleNodeConfig = "relu", output_activation: FlexibleNodeConfig = "num", initializer: Optional[Initializer] = None, learning_rate: float = 0.001, training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic", batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_epochs: int = 1000)

MLP: MLP(nodes_structure: list[int], activation: FlexibleNodeConfig = "relu", output_activation: FlexibleNodeConfig = "num", initializer: Optional[Initializer] = None, learning_rate: float = 0.001, training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic", batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_epochs: int = 1000) (Identical signature to Dense, serving as the Multi-Layer Perceptron wrapper)

2.3 Layers

General Instancing

layers.BaseLayer(_num_inputs: int, layer_configuration: LayerConstruction, batch_size: int = 1, gpu_id: int = 0)

General Attributes

delta, a, z

General Methods

set_parameters(params: dict) -> None (Accepts keys _weights, _biases, _connection_mask)

set_connection_mask(connection_mask: np.ndarray) -> None

get_config() -> dict[str, Any]

Subclass Instantiation Differences

LinearLayer: Follows general instancing.

(Direct attribute mutation of layer weights/biases is forbidden. Always use set_parameters to ensure cross-device consistency.)

2.4 API Wrapper

General Instancing

wrappers.Wrapper(model: Union[BaseNetwork, type[BaseNetwork]], work_type: Literal["class", "reg"], data_transformer: Optional[DataTransformer] = None)

General Attributes

model, work_type, grid_search_results, best_score

General Methods

load_training(X_train: np.ndarray, y_train: np.ndarray) -> None

run_training(num_iterations: int, batch_size: int = None) -> None

classification_test_accuracy(X: np.ndarray, y: np.ndarray) -> tuple[dict, Any]

regression_test_accuracy(X: np.ndarray, y: np.ndarray) -> dict

save_model(path: str) -> None

load_model(path: str) -> None

2.5 Data Transformers

General Instancing

utilities.DataTransformer()

General Attributes

is_fitted (property) -> bool

General Methods

fit(data: np.ndarray) -> None

transform(data: np.ndarray) -> np.ndarray

inverse_transform(data: np.ndarray) -> np.ndarray

get_config() -> dict[str, float]

set_config(config: dict[str, float]) -> None

2.6 Optimizers

General Instancing

optimizers.Optimizer(learning_rate: float = None, computational_device: Literal["GPU", "CPU"] = None, device_id: int = None)

General Attributes

learning_rate, CURRENT_DEVICE, COMPUTACIONAL_DEVICE

General Methods

get_config() -> dict[str, Any]

Subclass Instantiation Differences

Adam: Inherits general instantiation plus beta1, beta2, epsilon properties natively loaded from state.

SGD: Inherits general instantiation.

2.7 Losses

General Instancing

losses.Loss()

General Attributes

None

General Methods

forward(y_pred: BackendArray, y_true: BackendArray) -> BackendArray

backward(y_pred: BackendArray, y_true: BackendArray) -> BackendArray

set_gpu_id(new_id: int) -> None

Subclass Instantiation Differences

FlexibleLoss (Main class for custom losses): losses.FlexibleLoss(formula: str, constants: dict[str, float] = None, computational_method: Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None, gpu_id: int = 0) (Accepts custom SymPy strings)

HuberLoss: losses.HuberLoss(delta: float = 1.0, computational_method: Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None, gpu_id: int = 0)

MeanSquaredError / BinaryCrossEntropy / CategoricalCrossEntropy: Inherits general instancing with optional computational_method and gpu_id mapping.

3. Exception Handling

When writing robust code, wrap operations in the following custom exceptions from HeteroSymNN.exceptions:

HeteroSymNNError: Base exception.

BackendNotAvailableError / MethodMigrationError: Hardware routing failures.

CompilationWarning / JITError: Errors when SymPy string fails to compile to C++/CUDA.

ShapeMismatchError / LayerConfigurationError: Network architecture dimension issues.

WrapperError / LoadingError / SavingError: Issues inside the Wrapper API during disk I/O.

RuntimeStateError: Raised if execution methods are called out of order (e.g., calling transform before fit, backward pass before forward pass).