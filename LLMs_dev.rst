<system_directives>
- Math: Activations/losses MUST be valid SymPy strings (see LLMs_coding.rst <symbolic_formulas>).
- Conventions: Classes use PascalCase (`Wrapper`); module files use snake_case (`wrappers.py`); package folders use PascalCase (`Core`, `API`, `JIT`, `Backend`, `Nets`). Do not "correct" this.
- Devices: Networks, layers and losses use `"CPU_PYTHON"`/`"GPU_CUDA"` (computational methods). Optimizers use `"CPU"`/`"GPU"` exclusively (no backend kernels).
- Parser Vars: Accepted input vars: `"num"`, `"x"`, `"z"` (max 1 per expression). Normalized internally to `"num"`. Prefer `"num"`.
- Precision: `float32` only (`settings.set_precision` is not implemented yet).
</system_directives>

<forbidden_actions>
NEVER:
- Import External DL Frameworks: NO `torch`, `tensorflow`, `keras`. Use ONLY `numpy`, `cupy`, `sympy`.
- Write manual device-transfers: NO `.get()`/`.set()`. Use the backend module of the component (see <components_intro>) and `cast_arrays` / `asnumpy`.
- Mutate layer weights/biases directly from outside a layer: ALWAYS use `set_parameters(params)`/`get_parameters()`.
- Use "CPU_JIT" mode: It is disabled (falls back to "CPU_PYTHON").
- Instantiate Abstract Bases: Strictly subclass: `Initializer`, `BaseInitializer`, `DataTransformer`, `Optimizer`, `Loss`, `BaseLayer`.
</forbidden_actions>

<directory_structure>
/HeteroSymNN
├── __init__.py, __main__.py, cli.py, types.py, config.py, exceptions.py, error_handlers.py, utility.py
├── /Core: layers.py, losses.py, initializers.py, optimizers.py, legacy_core_objects.py
│   └── /Nets: base_classes.py, linear_net.py, legacy_nets.py, evo.py, recurrent.py
├── /JIT: compiler.py, codegen.py, validator.py (formula allowlist, runs before SymPy)
├── /Backend: hardware.py, validators.py, tuner.py
└── /API: registries.py, wrappers.py, data_transformers.py
Placeholders (not implemented yet): Nets/evo.py, Nets/recurrent.py, Backend/tuner.py.
</directory_structure>

<type_definitions>
- `NodeConfig`: `tuple[str, dict[str, float]]`
- `FlexibleNodeConfig`: `Union[str, NodeConfig]`
- `LayerValues`: `tuple[list[float], list[list[float]], list[list[float]]]` (Biases, Weights, ConnectionMask)
- `LayerConstruction`: `tuple[list[NodeConfig], Initializer]`
- `BackendArray`: `np.ndarray | cp.ndarray` (NumPy or CuPy)
- `ConstantToUpdate`: `tuple[int, str, float]` (Node Index, Constant Name, Constant Value)
</type_definitions>

<components_intro>
Strict instantiation/subclassing signatures. Type preservation (`BackendArray`) is CRITICAL for GPU compatibility: always create arrays with the component's backend module, never with `np` directly, inside code that runs on the active device.
Backend module per component:
- Losses and Optimizers: `self.be` (numpy or cupy) and `self.asnumpy(...)`.
- Layers and Networks: `self._CALCULATION_MANAGER` (numpy or cupy); public helpers `cast_arrays(...)` and `asnumpy(...)`.
</components_intro>

<networks>
# Execution Order & Data Flow:
# 1. predict / Wrapper: receives (samples, features) NumPy arrays, transposes them to (features, samples) and casts them to BackendArray.
# 2. forward passes the BackendArray sequentially through all layers; backward runs them in reverse.
# 3. predict returns (samples, outputs) as np.ndarray (if to_cpu=True).
```python
# Subclasses: HeteroLinearNet, LinearNet, MLP
class CustomNetwork(BaseNetwork):
    def __init__(self, num_inputs: int, network_structure: list[type[BaseLayer]], extra_layer_parameters: list[dict[str, Any]], detailed_activations: list[list[NodeConfig]], initializers: Optional[list[Initializer]]=None, learning_rate: float=0.001, batch_size: int=32, loss_function: Optional[Loss]=None, optimizer: Optional[Optimizer]=None, num_epochs: int=1000, gpu_id: int=0):
        super().__init__(num_inputs, network_structure, extra_layer_parameters, detailed_activations, initializers, learning_rate, batch_size, loss_function, optimizer, num_epochs, gpu_id) # MUST CALL
    # Methods: forward, backward, update_params, train_step, predict, cast_arrays, asnumpy, get_parameters, set_parameters, change_constants, set_backend, to, set_gpu_id, save_model, load_model, load_state, get_config, from_config
    # RO_Attrs: layers, network_structure, gpu_id, batch_size, current_device, current_location, computational_method, optimizer, loss_function, initializer, input_gradient, total_parameters
    # RW_Attrs: learning_rate, history_losses, num_completed_train_iterations, num_completed_epochs, num_training_epochs
```
</networks>

<layers>
# Execution Order & Data Flow:
# 1. forward(input_values): receives (inputs, batch) BackendArray -> z = (W * mask) @ x + b -> activation kernel -> returns (nodes, batch).
# 2. backward(error_values): receives dLoss/da (nodes, batch) -> delta = error * f'(z) via the activation kernel -> stores gradients in working_gradients -> returns dLoss/dx for the previous layer.
#
# Gradient Rule:
# - Loss.backward already returns the gradient of the MEAN loss (divided by the number of elements).
#   Layers must therefore SUM over the batch axis. Averaging again scales every gradient by 1/batch_size.
#
# Parameter Rules:
# - The structural masks and the mathematical parameters (e.g., weights) MUST be saved as separate arrays.
# - The mask parameters must ALWAYS be dense arrays, not sparse matrix structures.
```python
# Subclasses: LinearLayer
class CustomLayer(BaseLayer):
    def __init__(self, num_inputs: int, layer_configuration: LayerConstruction, batch_size: int=1, gpu_id: int=0):
        super().__init__(num_inputs, layer_configuration, batch_size, gpu_id) # MUST CALL

    # Condensed Forward Example (LinearLayer)
    def forward(self, input_values: BackendArray) -> BackendArray:
        super().forward(input_values)  # MUST CALL: caches the input and resizes buffers to the batch size
        batch_size = self.z.shape[1]
        effective_weights = self._weights * self._connection_mask  # Separate dense arrays
        self._CALCULATION_MANAGER.dot(effective_weights, input_values, out=self.z)
        self.z += self._biases
        self._act_funcions_manager.forward_kernel(self.z, self.a, self._funcs_constats, self.param_offsets, self._num_nodes, batch_size)
        return self.a

    # Condensed Backward Example (LinearLayer)
    def backward(self, error_values: BackendArray) -> BackendArray:
        error_values = super().backward(error_values)  # MUST CALL: validates state, returns a contiguous array
        batch_size = self.z.shape[1]
        self._act_funcions_manager.backward_kernel(self.z, error_values, self.delta, self._funcs_constats, self.param_offsets, self._num_nodes, batch_size)
        self._grad_biases = self._CALCULATION_MANAGER.sum(self.delta, axis=1, keepdims=True)    # SUM, see Gradient Rule
        self._grad_weights = self._CALCULATION_MANAGER.dot(self.delta, self._cached_input.T)   # SUM, see Gradient Rule
        effective_weights = self._weights * self._connection_mask
        self._CALCULATION_MANAGER.dot(effective_weights.T, self.delta, out=self._prev_error_buffer)
        return self._prev_error_buffer

    def get_parameters(self) -> dict[str, np.ndarray]: ...
    def set_parameters(self, params: dict[str, np.ndarray]) -> None: ...
    def get_config(self) -> dict[str, Any]: ...
    @property
    def working_parameters(self) -> dict[str, BackendArray]: ...  # live arrays the optimizer updates in place, e.g. {"weights": W, "biases": b}
    @property
    def working_gradients(self) -> dict[str, BackendArray]: ...   # same keys as working_parameters
    @property
    def working_masks(self) -> dict[str, BackendArray]: ...       # subset of keys with a connection mask, e.g. {"weights": mask}
```
</layers>

<loss_functions>
# Execution Order & Data Flow:
# The loss object is the ONLY source of truth for the loss value and its scaling. Layers and the Wrapper trust it as-is.
# 1. forward(y_pred, y_true): receives (outputs, batch) BackendArrays -> returns the reduced loss as a scalar BackendArray (built-ins: MEAN over all elements).
# 2. backward(y_pred, y_true): returns the exact gradient of forward's value w.r.t. y_pred, same shape as y_pred (for a mean: already divided by y_pred.size).
# 3. combine_batches(batch_losses, batch_sizes): combines per-batch forward values into the epoch value reported by training loops.
#    The default weights batches by size (correct for a mean). Override it together with forward/backward when using another reduction.
```python
# Subclasses: FlexibleLoss (-> MSELoss, MAELoss, HuberLoss, BinaryCrossEntropy, CategoricalCrossEntropy)
class CustomLoss(Loss):
    def __init__(self, computational_method: Literal["GPU_CUDA","CPU_PYTHON"]=None, gpu_id: int=0):
        super().__init__(computational_method, gpu_id) # MUST CALL
    def forward(self, y_pred: BackendArray, y_true: BackendArray) -> BackendArray:
        return self.be.mean(self.be.log(self.be.cosh(y_pred - y_true)))
    def backward(self, y_pred: BackendArray, y_true: BackendArray) -> BackendArray:
        return self.be.tanh(y_pred - y_true) / y_pred.size
    # combine_batches: the inherited default is correct here because forward is a mean.

# Example with a SUM reduction: forward, backward AND combine_batches change together.
class SumSquaredError(Loss):
    def forward(self, y_pred, y_true): return self.be.sum((y_pred - y_true) ** 2)
    def backward(self, y_pred, y_true): return 2 * (y_pred - y_true)
    def combine_batches(self, batch_losses, batch_sizes): return sum(batch_losses)
```
</loss_functions>

<optimizers>
# Execution Order & Data Flow:
# 1. Triggered by the network (update_params) after backprop. Iterates through layers.
# 2. _single_update(layer, ...): reads the layer's working_gradients -> applies the update IN PLACE using self.be -> the layer's working_parameters change directly on the active device.
```python
# Subclasses: AdamOptimizer, SgdOptimizer
class CustomOptimizer(Optimizer):
    def __init__(self, learning_rate: float=None, computational_device: Literal["GPU","CPU"]=None, device_id: int=None):
        super().__init__(learning_rate, computational_device, device_id) # MUST CALL
    def _refresh_parameters(self, vector_format):
        super()._refresh_parameters(vector_format) # MUST CALL: moves learning_rate to the active device
    def _single_update(self, layer: BaseLayer, param_name: str, param: BackendArray, grad: BackendArray, mask: Union[float, BackendArray]=1.0):
        param -= self.learning_rate * grad * mask  # MUST update in place
    # Stateful optimizers also override get_config() and get_state() / set_state(state, be) so their buffers are saved.
```
</optimizers>

<initializers>
# Execution Order & Data Flow:
# 1. Called strictly on CPU during layer instantiation or structural changes.
# 2. Generates and returns np.ndarray values. The layer then pushes these to the active device as BackendArrays.
```python
# Subclasses: RandomNormal, RandomUniform, XavierUniform, XavierNormal, HeUniform, HeNormal, LecunNormal, Orthogonal (all via BaseInitializer)
class CustomInitializer(Initializer):
    def __init__(self, scale: float=0.05):
        super().__init__() # MUST CALL
        self.scale = scale
    def generate_binary_mask(self, shape: list[int]) -> np.ndarray: ...
    def generate_constant(self, shape: list[int], value: float=0.0) -> np.ndarray: ...
    def generate_from_distribution(self, shape: list[int], fan_in: int, fan_out: int) -> np.ndarray: ...
    def get_config(self) -> dict[str, Any]: ...
```
</initializers>

<data_transformers>
# Execution Order & Data Flow:
# 1. Called strictly on CPU by the Wrapper. Operates entirely on np.ndarray.
# 2. Extracts statistics during fit(), and scales/descales data in transform() and inverse_transform().
# 3. is_fitted is READ-ONLY. The base class stores it through get_config()/set_config().
```python
# Subclasses: MinMaxScaler, StandardScaler
class CustomDataTransformer(DataTransformer):
    def __init__(self): super().__init__() # MUST CALL
    def fit(self, data: np.ndarray): return super().fit(data)                          # MUST call super() to mark as fitted.
    def transform(self, data: np.ndarray) -> np.ndarray: super().transform(data); ...  # MUST call super() to check the fitted state.
    def inverse_transform(self, data: np.ndarray) -> np.ndarray: super().inverse_transform(data); ...
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()  # MUST extend the base config (it holds the fitted state)
        config.update({...})
        return config
    def set_config(self, config: dict[str, Any]) -> None:
        super().set_config(config)     # MUST CALL: restores the fitted state
        ...
```
</data_transformers>

<api_wrapper>
```python
class Wrapper:
    def __init__(self, model: BaseNetwork, work_type: Literal["class","reg"], input_transformer: Optional[DataTransformer]=None, output_transformer: Optional[DataTransformer]=None, shuffle_samples: bool=False):
    # Methods: fit, predict, test_accuracy, load_training, run_training, classification_test_accuracy, regression_test_accuracy, save_model, load_model, load_state
    # RO_Attrs: training_data, training_data_norm. RW_Attrs: model, input_transformer, output_transformer, work_type, model_name, shuffle_samples

class GridSearchManager:
    def __init__(self, template_wrapper: Wrapper, param_grid: dict[str, list[Any]], validation_split: float=0.2):
    # Methods: load_data, execute_search
```
</api_wrapper>

<registries>
# Execution Order & Data Flow:
# 1. Define custom class (e.g., CustomLoss, CustomLayer, CustomNetwork).
# 2. Import the global `registry` from `HeteroSymNN.API.registries`.
# 3. Call the appropriate `add_*` method with your class (NOT an instance) BEFORE saving/loading or cloning models.
```python
from HeteroSymNN.API.registries import registry

# Example: Registering a custom loss
class MyCustomLoss(Loss): ...
registry.add_loss_func(MyCustomLoss)

# Example: Registering a custom network
class MyCustomNet(BaseNetwork): ...
registry.add_net(MyCustomNet)

# Available Methods:
# registry.add_initializer(Class)
# registry.add_loss_func(Class)
# registry.add_optimizer(Class)
# registry.add_layer(Class)
# registry.add_net(Class)
# registry.add_data_transformer(Class)
```
</registries>

<executable_contract>
# Executable Developer Contract:
# A complete, working implementation of this developer contract is in the repository at:
# Examples/custom_extensions_template.py
#
# This script:
# 1. Shows minimal, clean, CPU/GPU-portable subclasses for:
#    - `Loss` (Log-Cosh Loss)
#    - `Optimizer` (SGD with L2 Weight Decay)
#    - `Initializer` (Constant Scale Initializer)
#    - `DataTransformer` (Value Clipping Transformer)
# 2. Demonstrates how to register these custom classes with the global `registry`.
# 3. Runs an end-to-end training, saving, and loading verification to check that the serialization system reconstructs the custom classes cleanly.
# 4. Serves as a copy-pasteable boilerplate for extending the framework.
# It runs as part of the test suite (tests/test_examples.py), so it always matches the current contract.
USE THIS INFORMATION WHEN ASKED FOR EXAMPLES OF IMPLEMENTATION.
</executable_contract>
