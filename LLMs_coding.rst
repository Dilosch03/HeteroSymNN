<project_identity>
HeteroSymNN v0.3.0: JIT-compiled Differentiable Compiler & Neuro-Symbolic engine (dynamic SymPy expressions, no static graphs).
</project_identity>

<capabilities>
Capabilities: Design Networks (LinearNet, MLP, HeteroLinearNet), Define Symbolic Math (custom SymPy activations/losses), Train (API.Wrapper or custom loops), Extend Core (Initializers, Optimizers, Losses), Zero-Recompile Tuning (set_constants).
</capabilities>

<forbidden_actions>
NEVER:
- Import External DL Frameworks: NO `torch`, `tensorflow`, `keras`. Use ONLY `numpy`, `cupy`, `sympy`.
- Write manual device-transfers: NO `.get()`/`.set()` unless bypassing `BackendArray`. Routing is automatic.
- Mutate layer weights/biases directly: ALWAYS use `set_parameters(params)`/`get_parameters()`.
- Use "CPU_JIT" mode: It is disabled.
- Instantiate Abstract Bases: Strictly subclass: `Initializer`, `BaseInitializer`, `DataTransformer`, `Optimizer`, `Loss`, `BaseLayer`.
</forbidden_actions>

<import_patterns>
```python
from HeteroSymNN.Core.Nets import HeteroLinearNet, LinearNet, MLP, BaseNetwork
from HeteroSymNN.API import Wrapper, MinMaxScaler
from HeteroSymNN.Core import losses, optimizers, initializers, layers
from HeteroSymNN.types import FlexibleNodeConfig, NodeConfig, BackendArray, ConstantToUpdate, LayerValues, LayerConstruction
from HeteroSymNN.config import settings
```
</import_patterns>

<symbolic_formulas>
Parsed to C++/CUDA. Accepted vars: "num", "x", "z" (prefer "num").
Accepted String Activations: `"relu"`, `"sigmoid"`, `"swish"`/`"SiLU"`, `"softplus"`, `"mish"`, `"gelu"`, `"linear"`.
Accepted String Losses: `"mse"`, `"mae"`, `"huber"`, `"bce"`.
</symbolic_formulas>

<type_definitions>
- `NodeConfig`: `tuple[str, dict[str, float]]` -> Format: `("activation_expression", {"constant_name": default_value})`. Example: `("tanh(num)*a", {"a": 2.0})`.
- `FlexibleNodeConfig`: `str | NodeConfig` -> Can be just the expression string `"relu"` OR a full `NodeConfig` tuple.
- `LayerValues`: `tuple[list[float], list[list[float]], list[list[float]]]` -> Format: `(biases_1d, weights_2d, connection_mask_2d)`. Used for explicitly setting pre-trained arrays during init.
- `LayerConstruction`: `tuple[list[NodeConfig], Initializer]` -> Format: `(nodes_activation_list, layer_initializer)`.
- `BackendArray`: `np.ndarray | cp.ndarray` -> Hardware-agnostic array. NumPy on CPU, CuPy on GPU. Always use `self.be` to interface with these.
- `ConstantToUpdate`: `tuple[int, str, float]` -> Format: `(node_index, "constant_name", new_value)`. Used in `change_constants()`.
</type_definitions>

<networks_api>
*BaseNetwork*
- `__init__(num_inputs: int, network_structure: list[type[BaseLayer]], extra_layer_parameters: list[dict[str, Any]], detailed_activations: list[list[NodeConfig]], initializers: Optional[list[Initializer]] = None, learning_rate: float = 0.001, batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_epochs: int = 1000, gpu_id: int = 0)`
- `train(training_inputs: list, training_targets: list, ...) -> list[float]`
- `predict(input_values: list, to_cpu: bool = True) -> Union[np.ndarray, BackendArray]`
- `get_parameters() -> dict`
- `set_parameters(params: dict) -> None`
- `change_device(device: Literal["CPU", "GPU"]) -> None`
- `set_gpu_id(new_id: int) -> None`
- `change_constants(new_constants: dict) -> None`
- `get_config() -> dict[str, Any]`

*HeteroLinearNet*
- `__init__(num_inputs: int, detailed_activations: list[list[NodeConfig]], initializer: Optional[list[Initializer]] = None, learning_rate: float = 0.001, batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_training_iter: int = 1000)`

*LinearNet*
- `__init__(nodes_structure: list[int], activation_config: list[FlexibleNodeConfig], initial_values: Optional[list[LayerValues]] = None, initializer: Optional[Union[Initializer, list[Initializer]]] = None, learning_rate: float = 0.001, training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic", batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_training_iter: int = 1000)`

*MLP*
- `__init__(nodes_structure: list[int], activation: FlexibleNodeConfig = "relu", output_activation: FlexibleNodeConfig = "num", initializer: Optional[Initializer] = None, learning_rate: float = 0.001, training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic", batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_training_iter: int = 1000)`
</networks_api>

<wrapper_api>
*Wrapper*
- `__init__(model: BaseNetwork, work_type: Literal["class","reg"], input_transformer: Optional[data_transformers.DataTransformer] = None, output_transformer: Optional[data_transformers.DataTransformer] = None)`
- `fit(X: list, y: list, epochs: int = None, ...) -> list[float]`
- `predict(data: list) -> np.ndarray`
- `test_accuracy(X: list, y: list) -> Union[dict, tuple]`
- `load_training(X: np.ndarray, y: np.ndarray) -> None`
- `run_training(num_iterations: int, batch_size: int = None) -> None`
- `classification_test_accuracy(X: np.ndarray, y: np.ndarray) -> tuple[dict, Any]`
- `regression_test_accuracy(X: np.ndarray, y: np.ndarray) -> dict`
- `save_model(path: str) -> None`
- `load_model(path: str) -> None`
</wrapper_api>

<transformers_api>
*DataTransformer (Base)*
- `fit(data: np.ndarray) -> None`
- `transform(data: np.ndarray) -> np.ndarray`
- `inverse_transform(data: np.ndarray) -> np.ndarray`
- `get_config() -> dict[str, float]`
- `set_config(config: dict[str, float]) -> None`

*Available PreprocessingTransformers:*
    - `MinMaxScaler`
    - `StandardScaler`
</transformers_api>

<optimizers_api>
*AdamOptimizer*
- `__init__(learning_rate: float = None, computational_device: Literal["GPU", "CPU"] = None, device_id: int = None, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8)`

*SgdOptimizer*
- `__init__(learning_rate: float = None, computational_device: Literal["GPU", "CPU"] = None, device_id: int = None)`
</optimizers_api>

<losses_api>
*FlexibleLoss*
- `__init__(loss_expression: str = "(y_pred - y_true)**2", constants: dict[str, float] = None, computational_method: Literal["GPU_CUDA","CPU_PYTHON"] = None, gpu_id: int = 0)`

*Standard Losses (MSELoss, MAELoss, BinaryCrossEntropy)*
- `__init__(computational_method: Literal["GPU_CUDA","CPU_PYTHON"] = None, gpu_id: int = 0)`

*HuberLoss*
- `__init__(delta: float = 1.0, computational_method: Literal["GPU_CUDA","CPU_PYTHON"] = None, gpu_id: int = 0)`
</losses_api>

<initializers_api>
*RandomNormal*
- `__init__(mean: float = 0.0, stddev: float = 0.05, connection_density: float = None)`

*RandomUniform*
- `__init__(min_val: float = -0.05, max_val: float = 0.05, connection_density: float = None)`

*Orthogonal*
- `__init__(gain: float = 1.0, connection_density: float = None)`

*Standard Initializers (XavierUniform, XavierNormal, HeUniform, HeNormal, LecunNormal)*
- `__init__(connection_density: float = None)`
</initializers_api>

<settings_and_exceptions>
`HeteroSymNN.config.settings`: `use_kernel_cache` (bool), `n_jobs` (int), `warning_level` ("ignore"|"warn"|"error"), `default_compute_method` ("GPU_CUDA"|"CPU_PYTHON"), `cpu_cache_dir` (Path).
Exceptions (`HeteroSymNN.exceptions`): `HeteroSymNNError` (base), `BackendNotAvailableError`/`MethodMigrationError` (routing fails), `CompilationWarning`/`JITError`/`JITCompilationError`/`FormulaParsingError` (SymPy compile fails), `ShapeMismatchError`/`LayerConfigurationError`/`NetworkStructureError` (dim/structural mismatch), `WrapperError`/`LoadingError`/`SavingError`/`TrainingError` (disk I/O & training), `RuntimeStateError` (bad execution order), `HeteroSymNNValueError`/`DeviceSelectionError`/`ComputationalMethodValueError` (invalid options).
</settings_and_exceptions>

<cli_usage>
HeteroSymNN provides a Command-Line Interface invoked via `python -m HeteroSymNN <command>`:
- `hardware`: View system hardware detection and active defaults.
- `defaults`: View or update framework defaults.
  - Set defaults (permanently saves to `settings.json`): `python -m HeteroSymNN defaults --set <property> --value <value>`. Valid properties: `compute` (GPU_CUDA, CPU_PYTHON, CPU_JIT), `threads` (int), `use-cache` (true/false), `warnings` (error, ignore, always, default, module, once), `cpu-cache` (path).
  - Manage cache: `--clear-cpu-cache` (clears compiled kernels), `--show-gpu-cache` (shows CuPy directory).
- `parse`: Test SymPy string parsing (e.g. `python -m HeteroSymNN parse "sin(num)"`).
  - Flags: `--show-parsed` (displays internal SymPy expression), `--show-derivative` (displays the computed symbolic derivative).
- `inspect`: Read metadata, topology, and dynamic constants from a `.symnn` file without initializing a model.
- `clone`: Copy architecture/topology from an existing `.symnn` to a new `.symnn` without trained weights.
</cli_usage>

<framework_examples>
```python
# API Training Example (Full Lifecycle)
from HeteroSymNN.Core.Nets import LinearNet
from HeteroSymNN.API import Wrapper
from HeteroSymNN.Core import optimizers, losses, initializers

# 1. Instancing
custom_init = initializers.HeNormal(connection_density=0.8)
custom_loss = losses.HuberLoss(delta=1.5)
custom_opt = optimizers.AdamOptimizer(learning_rate=0.005)

model = LinearNet(nodes_structure=[10, 25, 1], activation_config=["sin(num)", ("tanh(num)*a", {"a":2.0})], initializer=custom_init, loss_function=custom_loss, optimizer=custom_opt)

# 2. Wrapping
agent = Wrapper(model, work_type="reg")

# 3. Training
agent.load_training(X_train, y_train)
losses = agent.run_training(num_iterations=100, batch_size=32)

# 4. Saving
agent.save_model("my_dense_model")

# 5. Loading (Creates a new Wrapper instance from the archive)
loaded_agent = Wrapper.load("my_dense_model.symnn")
predictions = loaded_agent.predict(X_test)

# Explicit Heterogeneous Topology (Full Lifecycle)
from HeteroSymNN.Core.Nets.linear_net import HeteroLinearNet
from HeteroSymNN.API import Wrapper
from HeteroSymNN.API.data_transformers import MinMaxScaler
from HeteroSymNN.Core import optimizers, losses, initializers

# 1. Instancing
h_acts = [["sin(num)",{}], ["Max(0, num)",{}], ["exp(num*beta)", {"beta": -0.5}]]*4 # 12 hidden nodes
hetero_model = HeteroLinearNet(num_inputs=2, detailed_activations=[h_acts, [["num",{}]]], loss_function=losses.BinaryCrossEntropy(), optimizer=optimizers.SgdOptimizer(learning_rate=0.01))

# 2. Wrapping (with Data Transformers)
agent_hetero = Wrapper(hetero_model, work_type="class", input_transformer=MinMaxScaler())

# 3. Training
agent_hetero.fit(X_train, y_train_class, epochs=200, batch_size=16)

# 4. Saving
agent_hetero.save_model("my_hetero_model")

# 5. Loading
loaded_hetero_agent = Wrapper.load("my_hetero_model.symnn")
metrics, raw_counts = loaded_hetero_agent.test_accuracy(X_test, y_test)
```
</framework_examples>
