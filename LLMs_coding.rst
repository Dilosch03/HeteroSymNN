<project_identity>
HeteroSymNN v0.3.0: JIT-compiled Differentiable Compiler & Neuro-Symbolic engine (dynamic SymPy expressions, no static graphs). Fully-connected networks where every neuron can have its own symbolic activation function.
</project_identity>

<capabilities>
Capabilities: Design Networks (LinearNet, MLP, HeteroLinearNet), Define Symbolic Math (custom SymPy activations/losses), Train (API.Wrapper or custom loops), Tune hyperparameters (API.GridSearchManager), Extend Core (Initializers, Optimizers, Losses, DataTransformers), Zero-Recompile Tuning (change_constants), Save/Load (.symnn archives).
</capabilities>

<forbidden_actions>
NEVER:
- Import External DL Frameworks: NO `torch`, `tensorflow`, `keras`. Use ONLY `numpy`, `cupy`, `sympy`.
- Write manual device-transfers: NO `.get()`/`.set()`. Use `model.cast_arrays(...)` to send data to the active backend and `model.asnumpy(...)` to bring it back.
- Mutate layer weights/biases directly: ALWAYS use `get_parameters()` / `set_parameters(params)`.
- Use "CPU_JIT" mode: It is disabled. Requesting it falls back to "CPU_PYTHON" with a `PerformanceWarning`.
- Instantiate Abstract Bases: Strictly subclass: `Initializer`, `BaseInitializer`, `DataTransformer`, `Optimizer`, `Loss`, `BaseLayer`.
- Build formulas that bypass the allowlist (e.g. calling SymPy directly on user strings): always go through the framework's classes, which validate formulas before SymPy parses them.
</forbidden_actions>

<preferred_actions>
WHEN CODING WITH HETEROSYMNN FOLLOW THESE PROTOCOLS:
- Set the batch size and number of training epochs (`num_training_iter`) at instantiation time so buffers are allocated once before training.
- Data layout: the `Wrapper`, `predict` and `train_step` take `(samples, features)`. The low-level primitives (`forward`, `backward`, `loss_function.forward/backward`) take `(features, samples)`.
- When writing a custom training loop with the network primitives (``forward``, ``loss_function.backward``, ``backward``, ``update_params``), increment the ``num_completed_train_iterations`` and ``num_completed_epochs`` counters yourself.
- In a custom training loop, send the whole dataset to the device once with ``model.cast_arrays(X, y)`` (already transposed to ``(features, samples)``) and slice batches from it, to avoid a host-to-device copy per batch.
</preferred_actions>

<import_patterns>
```python
from HeteroSymNN.Core.Nets import HeteroLinearNet, LinearNet, MLP, BaseNetwork
from HeteroSymNN.API import Wrapper, GridSearchManager, MinMaxScaler, StandardScaler
from HeteroSymNN.Core import losses, optimizers, initializers, layers
from HeteroSymNN.types import FlexibleNodeConfig, NodeConfig, BackendArray, ConstantToUpdate, LayerValues, LayerConstruction
from HeteroSymNN.config import settings
```
</import_patterns>

<symbolic_formulas>
Compiled to CUDA (GPU_CUDA) or vectorized NumPy (CPU_PYTHON). Accepted input variables: "num", "x", "z" (prefer "num"; use only one of them per expression).
Accepted String Activations: `"relu"`, `"sigmoid"`, `"silu"`, `"softplus"`, `"mish"`, `"gelu"`, `"linear"`, `"leaky_relu"`, `"swish"`.
Accepted String Losses: `"mse"`, `"mae"`, `"huber"`, `"bce"`.
Aliases are case-sensitive and lowercase: `"silu"` works, `"SiLU"` and `"MSE"` do not.

**Variable vs. Function Quirks (Golden Rule & Alias Exception):**
- **Golden Rule**: Any word NOT called with parentheses is treated as a constant that must be supplied in the constants dictionary (e.g., in `"sin * num"`, `sin` is a constant, so compilation fails unless `{"sin": ...}` is given). Any word called with parentheses is a mathematical function (e.g., `"sin(num)"`).
- **Alias Exception**: If a recognized alias (like `"relu"`, `"sigmoid"`) is the *only* text in the string, it is expanded to its functional form.
- **Parameterized Aliases**: `"leaky_relu"` requires `alpha` and `"swish"` requires `beta`. Default values are NOT injected yet, so always pass them: `("leaky_relu", {"alpha": 0.01})`, `("swish", {"beta": 1.0})`.
- **Composition**: Inside a larger equation, call aliases as functions (e.g., `"relu(num) * alfa"`); otherwise the alias is parsed as a constant.
- **Reserved names**: `params`, `offset`, `num`, `y_pred`, `y_true`, `z_val`, `e`, `pi`, `tau`, `phi` cannot be used as constant names. `e`, `pi`, `tau` and `phi` are the math constants.
- **Unsupported**: formulas whose derivative SymPy cannot compute, or that produce complex numbers, infinities or NaN, raise `FormulaParsingError`. Every node is checked and all errors are reported together.

**Allowed grammar (checked before SymPy sees the string, see `JIT/validator.py`):**
- Supported functions: `sin`, `cos`, `tan`, `sec`, `csc`, `cot`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `log`, `sqrt`, `cbrt`, `Abs`, `Max`, `Min`, `erf`, `erfc`, `Piecewise`, plus every alias above called as a function (e.g. `relu(num)`). Calls can be nested: `"sin(tanh(softplus(num)))"`.
- Operators: `+ - * / **`, unary `-`, comparisons `< <= > >=` and `&` / `|` for `Piecewise` conditions. Numbers, `True`/`False`, and ASCII names not starting with `_`.
- Rejected: attribute access, indexing, strings, lambdas, comprehensions, keyword arguments, `%`, `//`, `==`, `and`/`or`, `if/else` (use `Piecewise`), and any function not listed.
- Limits: 2000 characters, 500 elements, nesting depth 32, constant exponents up to 64 in constant-only powers (no `9**9**9`).
- Constant names must be valid ASCII identifiers not starting with `_`; constant values must be real numbers.
</symbolic_formulas>

<type_definitions>
- `NodeConfig`: `tuple[str, dict[str, float]]` -> Format: `("activation_expression", {"constant_name": value})`. Example: `("tanh(num)*a", {"a": 2.0})`.
- `FlexibleNodeConfig`: `str | NodeConfig` -> Can be just the expression string `"relu"` OR a full `NodeConfig` tuple.
- `LayerValues`: `tuple[list[float], list[list[float]], list[list[float]]]` -> Format: `(biases_1d, weights_2d, connection_mask_2d)`.
- `LayerConstruction`: `tuple[list[NodeConfig], Initializer]` -> Format: `(nodes_activation_list, layer_initializer)`.
- `BackendArray`: `np.ndarray | cp.ndarray` -> Hardware-agnostic array. NumPy on CPU, CuPy on GPU.
- `ConstantToUpdate`: `tuple[int, str, float]` -> Format: `(node_index, "constant_name", new_value)`. Used in `change_constants()`.
</type_definitions>

<networks_api>
*BaseNetwork*
- `__init__(num_inputs: int, network_structure: list[type[BaseLayer]], extra_layer_parameters: list[dict[str, Any]], detailed_activations: list[list[NodeConfig]], initializers: Optional[list[Initializer]] = None, learning_rate: float = 0.001, batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_epochs: int = 1000, gpu_id: int = 0)`
- `predict(input_values: list | np.ndarray, to_cpu: bool = True) -> np.ndarray | BackendArray` -> input `(samples, features)`, output `(samples, outputs)`.
- `forward(input_values: BackendArray) -> BackendArray` -> input `(features, samples)`.
- `backward(error_values: BackendArray) -> BackendArray`
- `update_params() -> None`
- `train_step(x_input: BackendArray, y_target: BackendArray) -> float` -> forward + loss + backward + update in one call. Takes `(samples, features)` and increments `num_completed_train_iterations` itself.
- `cast_arrays(*tensors: np.ndarray, gpu_id: int = None) -> BackendArray | tuple[BackendArray]`
- `asnumpy(*tensors: BackendArray) -> np.ndarray | tuple[np.ndarray]`
- `get_parameters() -> dict[str, dict[str, np.ndarray]]` -> `{"layer_0": {"weights": (nodes, inputs), "biases": (1, nodes), "connection_mask": (nodes, inputs)}, ...}`
- `set_parameters(params: dict[str, dict[str, np.ndarray]]) -> None`
- `change_constants(new_constants: dict[int, list[ConstantToUpdate] | ConstantToUpdate]) -> None` -> `{layer_index: [(node_index, "name", value), ...]}`
- `set_backend(backend: Literal["GPU_CUDA", "CPU_PYTHON"], gpu_id: int = None) -> None` -> recompiles kernels for the new backend.
- `to(location: Literal["host", "device"]) -> None` -> moves the arrays between CPU RAM and GPU VRAM.
- `set_gpu_id(new_id: int) -> None`
- `save_model(path: str, model_name: str = None, description: str = None, overwrite: bool = False, save_optimizer: bool = True) -> str`
- `load_model(path: str) -> BaseNetwork` -> classmethod, returns a new instance.
- `load_state(path: str) -> None` -> loads a saved state into this instance (topology must match).
- `get_config() -> dict[str, Any]`

*HeteroLinearNet*
- `__init__(num_inputs: int, detailed_activations: list[list[NodeConfig]], initializer: Optional[list[Initializer]] = None, learning_rate: float = 0.001, batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_training_iter: int = 1000)`

*LinearNet*
- `__init__(nodes_structure: list[int], activation_config: list[FlexibleNodeConfig], initializer: Optional[Initializer | list[Initializer]] = None, learning_rate: float = 0.001, batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_training_iter: int = 1000)` -> one `FlexibleNodeConfig` per layer (after the input), applied to every node of that layer.

*MLP*
- `__init__(nodes_structure: list[int], activation: FlexibleNodeConfig = "relu", output_activation: FlexibleNodeConfig = "num", initializer: Optional[Initializer] = None, learning_rate: float = 0.001, batch_size: int = 32, loss_function: Optional[Loss] = None, optimizer: Optional[Optimizer] = None, num_training_iter: int = 1000)`

Read-only attributes of every network: `layers`, `network_structure`, `gpu_id`, `batch_size`, `current_device`, `current_location`, `computational_method`, `optimizer`, `loss_function`, `initializer`, `input_gradient`, `total_parameters`. Plain attributes: `learning_rate` (settable), `history_losses`, `num_completed_train_iterations`, `num_completed_epochs`, `num_training_epochs`.
</networks_api>

<wrapper_api>
*Wrapper*
- `__init__(model: BaseNetwork, work_type: Literal["class", "reg"], input_transformer: Optional[DataTransformer] = None, output_transformer: Optional[DataTransformer] = None, shuffle_samples: bool = False)`
- `fit(training_data: list | np.ndarray, expected_results: list | np.ndarray, epochs: int = None, batch_size: int = None, return_batch_losses: bool = False) -> list[float]` -> returns the per-epoch loss history (or per-batch losses).
- `load_training(training_data: list | np.ndarray, expected_results: list | np.ndarray) -> None`
- `run_training(num_iterations: int = None, batch_size: int = None, return_batch_losses: bool = False) -> list[float]` -> trains on the data given to `load_training`.
- `predict(data: list | np.ndarray) -> np.ndarray`
- `test_accuracy(test_data: list, expected_results: list, threshold: float = 0.5) -> dict[str, float] | tuple[dict[str, float], dict[str, int]]` -> dict for "reg", `(metrics, confusion_counts)` for "class".
- `regression_test_accuracy(test_data: list, expected_results: list) -> dict[str, float]` -> keys include `"R2"`, `"MSE"`, `"RMSE"`.
- `classification_test_accuracy(test_data: list, expected_results: list, threshold: float = 0.5) -> tuple[dict[str, float], dict[str, int]]` -> metric keys include `"Acur"`.
- `save_model(path: str, model_name: str = None, description: str = None, overwrite: bool = False) -> str`
- `load_model(path: str) -> Wrapper` -> classmethod, returns a new Wrapper (model + transformers).
- `load_state(path: str) -> None`

*GridSearchManager*
- `__init__(template_wrapper: Wrapper, param_grid: dict[str, list[Any]], validation_split: float = 0.2)` -> grid keys: any key of the architecture config (`batch_size`, `num_training_epochs`, ...), the optimizer config (`learning_rate`, `beta1`, ...) or the loss config.
- `load_data(training_data: list, expected_results: list, shuffle: bool = True) -> None`
- `execute_search(metric_to_optimize: str = None, higher_is_better: bool = True) -> tuple[Wrapper, dict, list]` -> `(best_wrapper, best_params, all_results)`. Defaults to `"R2"` (reg) or `"Acur"` (class).
</wrapper_api>

<transformers_api>
*DataTransformer*
- `fit(data: np.ndarray) -> DataTransformer`
- `transform(data: np.ndarray) -> np.ndarray`
- `inverse_transform(data: np.ndarray) -> np.ndarray`
- `fit_transform(data: np.ndarray) -> np.ndarray`
- `get_config() -> dict[str, Any]`
- `set_config(config: dict[str, Any]) -> None`

*MinMaxScaler, StandardScaler*
- `__init__()`
</transformers_api>

<optimizers_api>
*AdamOptimizer*
- `__init__(learning_rate: float = None, computational_device: Literal["GPU", "CPU"] = None, device_id: int = None, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8)`

*SgdOptimizer*
- `__init__(learning_rate: float = None, computational_device: Literal["GPU", "CPU"] = None, device_id: int = None)`
</optimizers_api>

<losses_api>
Loss contract: the loss object is the ONLY source of truth for the loss value and its scaling. `forward` returns the reduced loss as a scalar (the built-in losses use the MEAN over all elements); `backward` returns the exact gradient of that value w.r.t. `y_pred` (same shape as `y_pred`, e.g. already divided by the number of elements for a mean). Layers never rescale gradients and the Wrapper never recomputes loss values: epoch losses come from `combine_batches`.

*Loss*
- `combine_batches(batch_losses: list[BackendArray], batch_sizes: list[int]) -> BackendArray` -> combines per-batch `forward` values into one epoch value. Default: weighted by batch size (correct for mean reductions). Override it if `forward` uses another reduction (e.g. a sum).

*FlexibleLoss*
- `__init__(loss_expression: str = "(y_pred - y_true)**2", constants: dict[str, float] = None, computational_method: Literal["GPU_CUDA", "CPU_PYTHON"] = None, gpu_id: int = 0)` -> expression uses `y_pred` and `y_true`, or a loss alias.
- `set_constants(constants: dict[str, float]) -> None`
- `get_constants() -> dict[str, float]`

*MSELoss, MAELoss, BinaryCrossEntropy, CategoricalCrossEntropy*
- `__init__(computational_method: Literal["GPU_CUDA", "CPU_PYTHON"] = None, gpu_id: int = 0)`

*HuberLoss*
- `__init__(delta: float = 1.0, computational_method: Literal["GPU_CUDA", "CPU_PYTHON"] = None, gpu_id: int = 0)`
</losses_api>

<initializers_api>
*RandomNormal*
- `__init__(mean: float = 0.0, stddev: float = 0.05, connection_density: float = None)`

*RandomUniform*
- `__init__(min_val: float = -0.05, max_val: float = 0.05, connection_density: float = None)`

*Orthogonal*
- `__init__(gain: float = 1.0, connection_density: float = None)`

*XavierUniform, XavierNormal, HeUniform, HeNormal, LecunNormal*
- `__init__(connection_density: float = None)`

`connection_density` (0-1] sets the fraction of active connections in the layer's binary connection mask. The default initializer is `HeNormal`.
</initializers_api>

<settings_and_exceptions>
`HeteroSymNN.config.settings` attributes: `use_kernel_cache` (bool), `n_jobs` (int), `warning_level` ("error" | "ignore" | "always" | "default" | "module" | "once"), `default_compute_method` ("GPU_CUDA" | "CPU_PYTHON"), `available_methods` (list), `cpu_cache_dir` (Path), `debug_mode` (bool, shows full tracebacks), `default_dtype` (float32 only).
Methods: `set_warning_level(value)`, `set_default_compute_method(method)`, `set_cache_location(path, move_existing_cache=False)`, `clear_kernel_cache(cache_type="ALL" | "CPU" | "GPU")`, `save()` (persists to `settings.json`).
Exceptions (`HeteroSymNN.exceptions`): `HeteroSymNNError` (base of all errors), `BackendNotAvailableError`/`MethodMigrationError` (routing fails), `CompilationWarning`/`JITError`/`JITCompilationError`/`FormulaParsingError` (formula or kernel compile fails), `ShapeMismatchError`/`LayerConfigurationError`/`NetworkStructureError` (dim/structural mismatch), `WrapperError`/`LoadingError`/`SavingError`/`TrainingError` (disk I/O & training), `RuntimeStateError` (bad execution order), `HeteroSymNNValueError`/`DeviceSelectionError`/`ComputationalMethodValueError`/`InvalidDeviceIDError` (invalid options), `PerformanceWarning`/`BackendNotAvailableWarning` (fallbacks).
</settings_and_exceptions>

<cli_usage>
HeteroSymNN provides a Command-Line Interface invoked via `python -m HeteroSymNN <command>` (or `heterosymnn <command>`):
- `version`: Print the installed framework version.
- `hardware`: View system hardware detection and active defaults.
- `defaults`: View or update framework defaults.
  - Set defaults (permanently saves to `settings.json`): `python -m HeteroSymNN defaults --set <property> --value <value>`. Valid properties: `compute` (GPU_CUDA, CPU_PYTHON, CPU_JIT), `threads` (int), `use-cache` (true/false), `warnings` (error, ignore, always, default, module, once), `cpu-cache` (path).
  - Manage cache: `--clear-cpu-cache` (clears compiled kernels), `--show-gpu-cache` (shows CuPy directory).
- `parse`: Test one or more formula strings (e.g. `python -m HeteroSymNN parse "sin(num)" "tanh(num)*a"`).
  - Flags: `--show-parsed` (displays the SymPy expression), `--show-derivative` (displays the symbolic derivative).
- `inspect`: Read metadata, topology, and dynamic constants from a `.symnn` file without initializing a model.
- `clone`: Copy the architecture of a `.symnn` into a new `.symnn` with fresh weights: `python -m HeteroSymNN clone source.symnn out.symnn`.
  - Flags: `--name`, `--description`, `--custom-scripts <paths>` (scripts that register custom classes before cloning).
</cli_usage>

<framework_examples>
```python
# API Training Example (Full Lifecycle). X_train: (samples, 10), y_train: (samples, 1)
from HeteroSymNN.Core.Nets import LinearNet
from HeteroSymNN.API import Wrapper
from HeteroSymNN.Core import optimizers, losses, initializers

# 1. Instancing
custom_init = initializers.HeNormal(connection_density=0.8)
custom_loss = losses.HuberLoss(delta=1.5)
custom_opt = optimizers.AdamOptimizer(learning_rate=0.005)

model = LinearNet(nodes_structure=[10, 25, 1], activation_config=["sin(num)", ("tanh(num)*a", {"a": 2.0})], initializer=custom_init, loss_function=custom_loss, optimizer=custom_opt, batch_size=32)

# 2. Wrapping
agent = Wrapper(model, work_type="reg")

# 3. Training
agent.load_training(X_train, y_train)
history = agent.run_training(num_iterations=100, batch_size=32)

# 4. Saving (".symnn" is appended if missing)
agent.save_model("my_dense_model", overwrite=True)

# 5. Loading (Creates a new Wrapper instance from the archive)
loaded_agent = Wrapper.load_model("my_dense_model.symnn")
predictions = loaded_agent.predict(X_test)

# Explicit Heterogeneous Topology (Full Lifecycle). y_train_class: (samples, 1) with 0/1 labels
from HeteroSymNN.Core.Nets import HeteroLinearNet
from HeteroSymNN.API import MinMaxScaler

# 1. Instancing: 12 hidden nodes with 3 different functions, sigmoid output for binary cross-entropy
h_acts = [("sin(num)", {}), ("Max(0, num)", {}), ("exp(num*beta)", {"beta": -0.5})] * 4
hetero_model = HeteroLinearNet(num_inputs=10, detailed_activations=[h_acts, [("sigmoid", {})]], loss_function=losses.BinaryCrossEntropy(), optimizer=optimizers.AdamOptimizer(learning_rate=0.01), batch_size=16)

# 2. Wrapping (with Data Transformers)
agent_hetero = Wrapper(hetero_model, work_type="class", input_transformer=MinMaxScaler())

# 3. Training
agent_hetero.fit(X_train, y_train_class, epochs=200, batch_size=16)

# 4. Zero-recompile tuning: layer 0, node 2 gets beta = -0.9
hetero_model.change_constants({0: [(2, "beta", -0.9)]})

# 5. Saving and loading
agent_hetero.save_model("my_hetero_model", overwrite=True)
loaded_hetero_agent = Wrapper.load_model("my_hetero_model.symnn")
metrics, raw_counts = loaded_hetero_agent.test_accuracy(X_test, y_test_class)
```
</framework_examples>
