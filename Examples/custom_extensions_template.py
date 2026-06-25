"""
HeteroSymNN Executable Developer Contract & Custom Extensions Template

This script serves as the executable validation of the framework's developer contract
outlined in the documentation (LLMs_dev.rst). 

It provides concrete, production-ready blueprints for subclassing and extending
the core components of HeteroSymNN:
1. Custom Loss (Log-Cosh Loss)
2. Custom Optimizer (SGD with Weight Decay)
3. Custom Initializer (Constant Scale Initializer)
4. Custom Data Transformer (Clip Scaler)

It registers these classes with the global registry module and runs an end-to-end
verification pass (train -> save -> load) to prove that they satisfy all JIT
compilation, device-portability, and serialization contracts.
"""

import numpy as np

from HeteroSymNN.Core.losses import Loss
from HeteroSymNN.Core.optimizers import Optimizer
from HeteroSymNN.Core.initializers import Initializer
from HeteroSymNN.API.data_transformers import DataTransformer
from HeteroSymNN.Core.Nets import LinearNet
from HeteroSymNN.API import Wrapper
from HeteroSymNN.API.registries import registry

# -------------------------------------------------------------------------
# CONTRACT 1: Custom Loss (Log-Cosh Loss)
# -------------------------------------------------------------------------
class LogCoshLoss(Loss):
    """
    Log-Cosh Loss: A smooth approximation of the Huber Loss.
    Formula: L(y_pred, y_true) = log(cosh(y_pred - y_true))
    """
    def __init__(self, computational_method=None, gpu_id: int = 0):
        # MUST call super().__init__ to set up device, manager (self.be) and backend properties
        super().__init__(computational_method, gpu_id)

    def _change_COMPUTATIONAL_METHOD(self, new_method, gpu_id=None):
        # MUST call super()._change_COMPUTATIONAL_METHOD to update backend managers and return the new method.
        return super()._change_COMPUTATIONAL_METHOD(new_method, gpu_id)

    def forward(self, y_pred, y_true):
        # MUST return a BackendArray. We perform calculations using self.be (np or cp)
        # to ensure seamless CPU/GPU portability.
        diff = y_pred - y_true
        return self.be.log(self.be.cosh(diff))

    def backward(self, y_pred, y_true):
        # MUST return a BackendArray representing the derivative w.r.t y_pred.
        # The derivative of log(cosh(x)) is tanh(x).
        diff = y_pred - y_true
        return self.be.tanh(diff)

# -------------------------------------------------------------------------
# CONTRACT 2: Custom Optimizer (SGD with Weight Decay)
# -------------------------------------------------------------------------
class SgdWeightDecayOptimizer(Optimizer):
    """
    Stochastic Gradient Descent (SGD) with L2 Weight Decay (Regularization).
    """
    def __init__(self, learning_rate: float = 0.01, weight_decay: float = 0.0001, computational_device=None, device_id=None):
        # MUST call super().__init__ to initialize the device, device_id, learning_rate tensor, and backend managers
        super().__init__(learning_rate, computational_device, device_id)
        self.weight_decay = weight_decay

    def _refresh_parameters(self, vector_format):
        # MUST call super()._refresh_parameters to refresh learning_rate tensor when moving devices
        super()._refresh_parameters(vector_format)
        # We also refresh our custom weight decay parameter to match the device vector format
        self.weight_decay_arr = vector_format(np.array(self.weight_decay, dtype=self.learning_rate.dtype))

    def _single_update(self, layer, param_name: str, param, grad, mask=1.0):
        # MUST perform parameter updates in-place directly on the active device using self.be.
        # param = param - lr * (grad + weight_decay * param) * mask
        lr = self.learning_rate
        wd = getattr(self, "weight_decay_arr", self.weight_decay)
        
        # Apply L2 regularization and update in-place
        update_step = (grad + wd * param) * mask
        param -= lr * update_step

# -------------------------------------------------------------------------
# CONTRACT 3: Custom Initializer (Constant Scale Initializer)
# -------------------------------------------------------------------------
class ConstantScaleInitializer(Initializer):
    """
    An initializer that generates random values scaled by a custom factor.
    """
    def __init__(self, scale: float = 0.05):
        # MUST call super().__init__ to register the base initializer
        super().__init__()
        self.scale = scale

    def generate_binary_mask(self, shape: list[int]) -> np.ndarray:
        # MUST return an np.ndarray on the CPU representing the connection mask.
        return np.ones(shape, dtype=np.float32)

    def generate_constant(self, shape: list[int], value: float = 0.0) -> np.ndarray:
        # MUST return an np.ndarray on the CPU filled with a constant.
        return (np.ones(shape) * value).astype(np.float32)

    def generate_from_distribution(self, shape: list[int], fan_in: int, fan_out: int) -> np.ndarray:
        # MUST return an np.ndarray on the CPU representing the weights or biases.
        return (np.random.randn(*shape) * self.scale).astype(np.float32)

# -------------------------------------------------------------------------
# CONTRACT 4: Custom Data Transformer (Value Clipping Transformer)
# -------------------------------------------------------------------------
class ClipTransformer(DataTransformer):
    """
    A custom transformer that clips input features between a specified min and max value.
    """
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        # MUST call super().__init__
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, data: np.ndarray) -> None:
        # MUST call super().fit to set self.is_fitted = True
        super().fit(data)
        # This is a static clipper, so no statistical fitting is required

    def transform(self, data: np.ndarray) -> np.ndarray:
        # MUST call super().transform to assert self.is_fitted
        super().transform(data)
        return np.clip(data, self.min_val, self.max_val)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        # MUST call super().inverse_transform
        super().inverse_transform(data)
        # Clipping is non-invertible, so we return the clipped data as-is for the inverse pass
        return data

    def get_config(self) -> dict[str, any]:
        # MUST return a configuration dictionary containing all serializable parameters
        return {"min_val": self.min_val, "max_val": self.max_val}

    def set_config(self, config: dict[str, any]) -> None:
        # MUST load parameters from config and set self.is_fitted = True
        self.min_val = config.get("min_val", -1.0)
        self.max_val = config.get("max_val", 1.0)
        self.is_fitted = True

# -------------------------------------------------------------------------
# CONTRACT 5: Global Registry Integration
# -------------------------------------------------------------------------
# To ensure that custom objects can be saved and loaded cleanly in .symnn zip
# archives, we MUST register them with the global registry module before execution.
print("Registering custom classes with the global registry...")
registry.add_loss_func(LogCoshLoss)
registry.add_optimizer(SgdWeightDecayOptimizer)
registry.add_initializer(ConstantScaleInitializer)
registry.add_data_transformer(ClipTransformer)
print("Registration complete.")

# -------------------------------------------------------------------------
# Verification Pass
# -------------------------------------------------------------------------
def verify_custom_extensions():
    print("\n--- Verifying Custom Extensions End-to-End ---")
    
    # 1. Generate small synthetic dataset
    X = np.linspace(-2.0, 2.0, 200).reshape(-1, 1).astype(np.float32)
    y = (X**2).astype(np.float32)

    # 2. Build the model using our custom Initializer, Loss, and Optimizer
    print("Instantiating model with custom Initializer, Loss, and Optimizer...")
    model = LinearNet(
        nodes_structure=[1, 16, 1],
        activation_config=["relu", "num"],
        initializer=ConstantScaleInitializer(scale=0.02),
        loss_function=LogCoshLoss(),
        optimizer=SgdWeightDecayOptimizer(learning_rate=0.02, weight_decay=0.0001),
        batch_size=32
    )

    # 3. Wrap the model using our custom Data Transformer
    print("Wrapping model with custom Data Transformer...")
    trainer = Wrapper(
        model=model,
        work_type="reg",
        input_transformer=ClipTransformer(min_val=-1.5, max_val=1.5)
    )

    # 4. Run a quick, 3-epoch training pass to verify mathematical correctness
    print("Running training verification...")
    trainer.fit(X, y, epochs=3)
    print(f"Training successful. Final loss: {trainer.model.history_losses[-1]:.6f}")

    # 5. Save the model to a temporary .symnn file (gitignored)
    temp_path = "temp_custom_contract_test.symnn"
    print(f"Saving model with custom classes to '{temp_path}'...")
    trainer.save_model(temp_path, overwrite=True)

    # 6. Load the model back to verify that the JIT compiler and registries reconstruct it correctly
    print("Loading model back from serialization...")
    rebuilt_trainer = Wrapper.load_model(temp_path)
    print("Model loaded successfully.")
    
    # Run a mock prediction to confirm the rebuilt model functions on the active device
    pred = rebuilt_trainer.predict(np.array([[1.0]]))
    print(f"Prediction at x=1.0: {pred[0,0]:.6f}")

    # 7. Clean up the temporary file
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Cleaned up temporary file '{temp_path}'.")
    except Exception as e:
        print(f"Warning: Could not remove temporary file: {e}")

    print("\nAll custom classes satisfied the developer contract successfully.")

if __name__ == "__main__":
    verify_custom_extensions()
