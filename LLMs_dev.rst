<system_directives>
- Math: Activations/losses MUST be valid SymPy strings.
- Conventions: Classes use PascalCase (`Wrapper`); files use snake_case (`wrapper.py`). Do not "correct" this.
- Devices: Most components use `"CPU_PYTHON"`/`"GPU_CUDA"`. Optimizers use `"CPU"`/`"GPU"` exclusively (no backend kernels).
- Parser Vars: Accepted input vars: `"num"`, `"x"`, `"z"` (max 1 per expression). Normalized internally to `"num"`. Prefer `"num"`.
</system_directives>

<forbidden_actions>
NEVER:
- Import External DL Frameworks: NO `torch`, `tensorflow`, `keras`. Use ONLY `numpy`, `cupy`, `sympy`.
- Write manual device-transfers: NO `.get()`/`.set()` unless bypassing `BackendArray`. Routing is automatic.
- Mutate layer weights/biases directly: ALWAYS use `set_parameters(params)`/`get_parameters()`.
- Use "CPU_JIT" mode: It is disabled.
- Instantiate Abstract Bases: Strictly subclass: `Initializer`, `BaseInitializer`, `DataTransformer`, `Optimizer`, `Loss`, `BaseLayer`.
</forbidden_actions>

<directory_structure>
/HeteroSymNN
├── types.py, config.py, exceptions.py
├── /Core: layers.py, losses.py, initializers.py, optimizers.py, /Nets (base.py, dense.py, evo.py, functional.py)
├── /JIT: compiler.py, codegen.py
├── /Backends: hardware.py
└── /API: registries.py, wrappers.py, data_transformers.py
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
Strict instantiation/subclassing signatures. Type preservation (`BackendArray`) is CRITICAL for GPU compatibility.
Properties for subclasses: `self.be` (Backend: numpy/cupy. Read-Only. Nets, Layers, Optimizers), `self.asnumpy` (Convert to NumPy. Read-Only. Nets, Layers. AVOID unless exclusively required).
</components_intro>

<networks>
# Execution Order & Data Flow:
# 1. train/predict: Receives NumPy arrays from Wrapper/User. Internally converts to BackendArray via computation manager.
# 2. Passes BackendArray sequentially through all layers.
# 3. Returns predictions as np.ndarray (if to_cpu=True).
```python
# Subclasses: HeteroDense, Dense, MLP
class CustomNetwork(BaseNetwork):
    def __init__(self, network_structure: list[tuple[int, type[BaseLayer]]], extra_layer_parameters: list[dict[str, Any]], detailed_activations: list[list[NodeConfig]], initial_values: Optional[list[LayerValues]]=None, initializers: Optional[list[Initializer]]=None, learning_rate: float=0.001, batch_size: int=32, training_mode: Literal["batch","mini-batch","stochastic"]="mini-batch", loss_function: Optional[Loss]=None, optimizer: Optional[Optimizer]=None, num_epochs: int=1000):
        super().__init__(...) # MUST CALL
    # Methods: train, predict, get_parameters, set_parameters, change_device, set_gpu_id, change_constants, get_config
    # RO_Attrs: layers, network_structure, gpu_id, batch_size, current_device, computational_method, optimizer, loss_function, initializer, history_losses, num_completed_train_iterations, num_completed_epochs
    # RW_Attrs: learning_rate, training_mode, num_training_epochs
```
</networks>

<layers>
# Execution Order & Data Flow:
# 1. forward(input_values): Receives BackendArray from previous layer -> multiplies with working_parameters (BackendArray) -> applies activation -> returns BackendArray to next layer.
# 2. backward(error_values): Receives BackendArray (error) from next layer -> computes local gradients -> saves to working_gradients -> passes error BackendArray to previous layer.
# 
# Parameter Rules: 
# - The structural masks and the mathematical parameters (e.g., weights) MUST be saved as separate arrays.
# - The mask parameters must ALWAYS be dense arrays, not sparse matrix structures.
```python
# Subclasses: LinearLayer
class CustomLayer(BaseLayer):
    def __init__(self, num_inputs: int, layer_configuration: LayerConstruction, batch_size: int=1, gpu_id: int=0):
        super().__init__(num_inputs, layer_configuration, batch_size, gpu_id) # MUST CALL

    # Resumed Forward Example (LinearLayer)
    def forward(self, input_values: BackendArray) -> BackendArray:
        self._cached_input = input_values
        effective_weights = self._weights * self._connection_mask # Separated, dense arrays
        self.z = self._CALCULATION_MANAGER.dot(effective_weights, input_values) + self._biases
        # ... Apply activation via self._act_funcions_manager ...
        return self.a

    # Resumed Backward Example (LinearLayer)
    def backward(self, error_values: BackendArray) -> BackendArray:
        # ... Calculate self.delta (activation derivatives) via self._act_funcions_manager ...
        self._grad_biases = self._CALCULATION_MANAGER.mean(self.delta, axis=1, keepdims=True)
        self._grad_weights = self._CALCULATION_MANAGER.dot(self.delta, self._cached_input.T) / batch_size
        effective_weights = self._weights * self._connection_mask
        prev_layer_error_sum = self._CALCULATION_MANAGER.dot(effective_weights.T, self.delta)
        return prev_layer_error_sum

    def get_parameters(self) -> dict[str, np.ndarray]: ...
    def set_parameters(self, params: dict[str, np.ndarray]) -> None: ...
    @property
    def working_parameters(self) -> dict[str, BackendArray]: ...
    @property
    def working_gradients(self) -> dict[str, BackendArray]: ...
    @property
    def working_masks(self) -> dict[str, BackendArray]: ...
```
</layers>

<loss_functions>
# Execution Order & Data Flow:
# 1. forward(y_pred, y_true): Receives BackendArrays from Network -> computes scalar loss -> returns BackendArray.
# 2. backward(y_pred, y_true): Computes derivative w.r.t predictions -> returns BackendArray to initialize backpropagation.
```python
# Subclasses: FlexibleLoss, MSELoss, MAELoss, HuberLoss, BinaryCrossEntropy
class CustomLoss(Loss):
    def __init__(self, computational_method: Literal["GPU_CUDA","CPU_PYTHON"]=None, gpu_id: int=0):
        super().__init__(computational_method, gpu_id) # MUST CALL
    def forward(self, y_pred: BackendArray, y_true: BackendArray) -> BackendArray: ... # MUST return BackendArray.
    def backward(self, y_pred: BackendArray, y_true: BackendArray) -> BackendArray: ... # MUST return BackendArray.
```
</loss_functions>

<optimizers>
# Execution Order & Data Flow:
# 1. Triggered by Network after backprop. Iterates through layers.
# 2. _single_update(layer, ...): Reads layer working_gradients -> applies update in-place using self.be -> updates layer working_parameters directly on active device.
```python
# Subclasses: AdamOptimizer, SgdOptimizer
class CustomOptimizer(Optimizer):
    def __init__(self, learning_rate: float=None, computational_device: Literal["GPU","CPU"]=None, device_id: int=None):
        super().__init__(learning_rate, computational_device, device_id) # MUST CALL
    def _refresh_parameters(self, vector_format): ...
    def _single_update(self, layer: BaseLayer, param_name: str, param: BackendArray, grad: BackendArray, mask: Union[float, BackendArray]=1.0): ...
```
</optimizers>

<initializers>
# Execution Order & Data Flow:
# 1. Called strictly on CPU during Layer instantiation or structural changes.
# 2. Generates and returns np.ndarray values. The layer then pushes these to the active device as BackendArrays.
```python
# Subclasses: RandomNormal, RandomUniform, XavierUniform, XavierNormal, HeUniform, HeNormal, LecunNormal, Orthogonal
class CustomInitializer(Initializer):
    def __init__(self, connection_density: float=1.0):
        super().__init__() # MUST CALL
    def generate_binary_mask(self, shape: list[int]) -> np.ndarray: ...
    def generate_constant(self, shape: list[int], value: float=0.0) -> np.ndarray: ...
    def generate_from_distribution(self, shape: list[int], fan_in: int, fan_out: int) -> np.ndarray: ...
```
</initializers>

<data_transformers>
# Execution Order & Data Flow:
# 1. Called strictly on CPU by Wrapper. Operates entirely on np.ndarray.
# 2. Extracts statistics during fit(), and scales/descales data in transform() and inverse_transform().
```python
# Subclasses: MinMaxScaler
class CustomDataTransformer(DataTransformer):
    def __init__(self): super().__init__() # MUST CALL
    def fit(self, data: np.ndarray) -> None: super().fit(data) # MUST call super() to set `is_fitted`.
    def transform(self, data: np.ndarray) -> np.ndarray: return super().transform(data) # MUST call super() to check `is_fitted`.
    def inverse_transform(self, data: np.ndarray) -> np.ndarray: return super().inverse_transform(data) # MUST call super().
    def get_config(self) -> dict[str, Any]: ...
    def set_config(self, config: dict[str, Any]) -> None: ...
```
</data_transformers>

<api_wrapper>
```python
class Wrapper:
    def __init__(self, model: BaseNetwork, work_type: Literal["class","reg"], input_transformer: Optional[DataTransformer]=None, output_transformer: Optional[DataTransformer]=None): ...
    # Methods: fit, predict, test_accuracy, load_training, run_training, classification_test_accuracy, regression_test_accuracy, save_model, load_model
    # RO_Attrs: training_data, training_data_norm. RW_Attrs: model, input_transformer, output_transformer, work_type, model_name
```
</api_wrapper>

<registries>
# Execution Order & Data Flow:
# 1. Define custom class (e.g., CustomLoss, CustomLayer, CustomNetwork).
# 2. Import the global `registry` from `HeteroSymNN.API.registries`.
# 3. Call the appropriate `add_*` method with your class pointer (NOT an instance) BEFORE loading/saving or executing models.
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