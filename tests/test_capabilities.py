import sys
import os
import time
import pytest
import numpy as np
import warnings

# Ensure HeteroSymNN is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HeteroSymNN.config import settings
from HeteroSymNN import exceptions
from HeteroSymNN.Core import losses, optimizers, initializers, layers
from HeteroSymNN.Core.Nets import HeteroLinearNet, LinearNet, MLP, BaseNetwork
from HeteroSymNN.API.wrappers import Wrapper
from HeteroSymNN.API.data_transformers import MinMaxScaler

# --- Global Dummies ---
X_dummy = np.random.rand(8, 4).astype(np.float32)
X_cls = np.random.rand(40, 4).astype(np.float32)
y_cls = np.random.randint(0, 2, (40, 1)).astype(np.float32)
X_reg = np.random.rand(40, 4).astype(np.float32)
y_reg = np.random.rand(40, 1).astype(np.float32)

class TestSettings:
    def test_settings_read(self):
        _ = settings.use_kernel_cache
        _ = settings.n_jobs
        _ = settings.warning_level
        _ = settings.default_compute_method
        _ = settings.cpu_cache_dir

    def test_settings_methods(self):
        settings.set_warning_level("once")
        settings.set_default_compute_method("CPU_PYTHON")
        settings.clear_kernel_cache("CPU")

class TestInitializers:
    @pytest.mark.parametrize("cls_name", [
        "RandomNormal", "RandomUniform", "XavierUniform", "XavierNormal",
        "HeUniform", "HeNormal", "LecunNormal", "Orthogonal"
    ])
    def test_initializers(self, cls_name):
        cls = getattr(initializers, cls_name)
        obj = cls()
        mask = obj.generate_binary_mask([4, 4])
        const = obj.generate_constant([4, 4], value=0.1)
        dist = obj.generate_from_distribution([4, 4], fan_in=4, fan_out=4)
        cfg = obj.get_config()
        assert isinstance(cfg, dict)

class TestLosses:
    _y_pred = np.array([0.8, 0.2, 0.6])
    _y_true = np.array([1.0, 0.0, 1.0])

    @pytest.mark.parametrize("cls_name", ["MSELoss", "MAELoss", "HuberLoss", "BinaryCrossEntropy"])
    def test_standard_losses(self, cls_name):
        cls = getattr(losses, cls_name)
        loss = cls()
        out = loss.forward(self._y_pred, self._y_true)
        grad = loss.backward(self._y_pred, self._y_true)
        cfg = loss.get_config()
        assert isinstance(cfg, dict)

    def test_flexible_loss_custom(self):
        loss = losses.FlexibleLoss("(y_pred - y_true)**2")
        out = loss.forward(self._y_pred, self._y_true)
        grad = loss.backward(self._y_pred, self._y_true)

    def test_flexible_loss_with_constants(self):
        loss = losses.FlexibleLoss(
            "Piecewise((0.5*(y_pred-y_true)**2, Abs(y_pred-y_true)<=delta), (delta*Abs(y_pred-y_true)-0.5*delta**2, True))",
            constants={"delta": 1.0},
        )
        loss.forward(self._y_pred, self._y_true)

class TestOptimizers:
    @pytest.mark.parametrize("cls_name", ["AdamOptimizer", "SgdOptimizer"])
    def test_optimizers(self, cls_name):
        cls = getattr(optimizers, cls_name)
        opt = cls(learning_rate=0.001)
        cfg = opt.get_config()
        assert isinstance(cfg, dict)

class TestNetworks:
    def test_dense_basic(self):
        model = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "sigmoid"], num_training_iter=1)
        out = model.predict(X_dummy)
        assert out.shape[0] == 8

    def test_dense_symbolic(self):
        model = LinearNet(nodes_structure=[4, 6, 1], activation_config=["sin(num)", ("tanh(num) * a", {"a": 2.0})], num_training_iter=1)
        model.predict(X_dummy)

    def test_mlp_basic(self):
        model = MLP(nodes_structure=[4, 8, 1], activation="relu", output_activation="num", num_training_iter=1)
        out = model.predict(X_dummy)
        assert out.shape[0] == 8

    def test_heterodense_basic(self):
        model = HeteroLinearNet(
            num_inputs=4,
            detailed_activations=[
                [("relu", {}), ("sigmoid", {}), ("relu", {}), ("sigmoid", {}), ("relu", {}), ("sigmoid", {})],
                [("linear", {})],
            ],
            num_training_iter=1,
        )
        model.predict(X_dummy)

    def test_network_get_config(self):
        model = LinearNet(nodes_structure=[4, 4, 1], activation_config=["relu", "linear"], num_training_iter=1)
        cfg = model.get_config()
        assert isinstance(cfg, dict)

    def test_network_forward_backward(self):
        model = LinearNet(nodes_structure=[4, 4, 1], activation_config=["relu", "sigmoid"], loss_function=losses.MSELoss(), num_training_iter=1)
        out = model.forward(X_dummy.T)
        y_t = np.random.rand(1, 8).astype(np.float32)
        error = out - y_t
        model.backward(error)

    def test_network_change_device(self):
        model = LinearNet(nodes_structure=[4, 4, 1], activation_config=["relu", "linear"], num_training_iter=1)
        model.to("CPU")

    def test_zero_recompile(self):
        model = LinearNet(nodes_structure=[4, 4, 1], activation_config=[("relu", {}), ("tanh(num) * alpha", {"alpha": 1.0})], num_training_iter=1)
        
        # Cold run: Forces initial C++/CUDA JIT compilation
        model.predict(X_dummy)
        
        # Dynamically inject the new constant
        model.change_constants({1: [(0, "alpha", 2.0)]})
        
        # Hot run: Should strictly bypass compilation and use cached kernel
        start_time = time.perf_counter()
        model.predict(X_dummy)
        hot_duration = time.perf_counter() - start_time
        
        # Compilation takes > 0.5s. If it executes under 0.1s, we prove nvcc wasn't called.
        assert hot_duration < 0.1, f"Zero-recompile failed, hot run took {hot_duration}s (likely recompiled)"

    def test_graceful_cpu_fallback(self, monkeypatch):
        # Hijack the available methods to simulate no GPU
        monkeypatch.setattr(settings, "_available_methods", ["CPU_PYTHON"])
        
        # Force the settings back to default warning level to ensure the warning fires
        settings.set_warning_level("default")
        
        # We expect the BackendNotAvailableWarning to trigger when it fails to initialize the backend
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Requesting GPU_CUDA explicitly should trigger the fallback sequence
            settings.set_default_compute_method("GPU_CUDA")
            model = LinearNet(nodes_structure=[4, 4, 1], activation_config=["relu", "linear"], num_training_iter=1)
            model.predict(X_dummy)
            
            # Verify the warning was properly caught and recorded
            fallback_warning_found = any(issubclass(warn.category, exceptions.BackendNotAvailableWarning) for warn in w)
            assert fallback_warning_found, "The framework did not emit a BackendNotAvailableWarning during GPU fallback."

    def test_network_save_load(self, tmp_path):
        model = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "linear"], num_training_iter=1)
        original_preds = model.predict(X_dummy)
        save_path = str(tmp_path / "raw_model")
        model.save_model(save_path)
        
        # Test class method load_model
        loaded_model = BaseNetwork.load_model(save_path)
        loaded_preds = loaded_model.predict(X_dummy)
        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5, atol=1e-5)
        assert loaded_model.get_config() == model.get_config()
        
        # Test instance method load_state
        model_new = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "linear"], num_training_iter=1)
        model_new.load_state(save_path)
        new_preds = model_new.predict(X_dummy)
        np.testing.assert_allclose(original_preds, new_preds, rtol=1e-5, atol=1e-5)
        assert model_new.get_config() == model.get_config()

class TestAPIWrapper:
    def test_wrapper_regression_fit(self):
        model = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "linear"], num_training_iter=2)
        agent = Wrapper(model, work_type="reg")
        agent.fit(X_reg.tolist(), y_reg.tolist(), epochs=2)

    def test_wrapper_classification_fit(self):
        model = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "sigmoid"], num_training_iter=2)
        agent = Wrapper(model, work_type="class")
        agent.fit(X_cls.tolist(), y_cls.tolist(), epochs=2)

    def test_wrapper_predict(self):
        model = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "linear"], num_training_iter=2)
        agent = Wrapper(model, work_type="reg")
        agent.fit(X_reg.tolist(), y_reg.tolist(), epochs=2)
        preds = agent.predict(X_reg[:5].tolist())
        assert preds is not None

    def test_wrapper_test_accuracy_reg(self):
        model = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "linear"], num_training_iter=2)
        agent = Wrapper(model, work_type="reg")
        agent.fit(X_reg.tolist(), y_reg.tolist(), epochs=2)
        agent.test_accuracy(X_reg.tolist(), y_reg.tolist())

    def test_wrapper_load_run_training(self):
        model = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "linear"], num_training_iter=2)
        agent = Wrapper(model, work_type="reg")
        agent.load_training(X_reg, y_reg)
        agent.run_training(num_iterations=2, batch_size=8)

    def test_wrapper_save_load(self, tmp_path):
        model = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "linear"], num_training_iter=2)
        agent = Wrapper(model, work_type="reg")
        agent.fit(X_reg.tolist(), y_reg.tolist(), epochs=2)
        save_path = str(tmp_path / "hetero_test_model")
        agent.save_model(save_path)

        # Get original values
        original_preds = agent.predict(X_reg.tolist())
        original_config = agent.model.get_config()
        original_params = agent.model.get_parameters()
        original_opt_config = agent.model.optimizer.get_config()

        # Test load_model (classmethod) - loads into a new wrapper instance
        loaded_agent = Wrapper.load_model(save_path)

        # Test load_state - loads state into the existing wrapper
        agent.load_state(save_path)

        # Verify new instance predictions match original predictions
        loaded_preds = loaded_agent.predict(X_reg.tolist())
        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5, atol=1e-5)

        # Verify state-restored instance predictions match original predictions
        state_preds = agent.predict(X_reg.tolist())
        np.testing.assert_allclose(original_preds, state_preds, rtol=1e-5, atol=1e-5)

        # Verify model configs are identical
        assert loaded_agent.model.get_config() == original_config
        assert agent.model.get_config() == original_config

        # Verify optimizer configurations match
        assert loaded_agent.model.optimizer.get_config() == original_opt_config
        assert agent.model.optimizer.get_config() == original_opt_config

        # Verify parameters (weights and biases) are identical
        loaded_params = loaded_agent.model.get_parameters()
        state_params = agent.model.get_parameters()
        for layer_key in original_params:
            for param_key in original_params[layer_key]:
                np.testing.assert_allclose(
                    original_params[layer_key][param_key],
                    loaded_params[layer_key][param_key],
                    rtol=1e-5, atol=1e-5
                )
                np.testing.assert_allclose(
                    original_params[layer_key][param_key],
                    state_params[layer_key][param_key],
                    rtol=1e-5, atol=1e-5
                )

    def test_wrapper_with_data_transformer(self):
        scaler = MinMaxScaler()
        model = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "linear"], num_training_iter=2)
        agent = Wrapper(model, work_type="reg", input_transformer=scaler)
        agent.fit(X_reg.tolist(), y_reg.tolist(), epochs=2)

class TestDataTransformers:
    def test_min_max_scaler(self):
        scaler = MinMaxScaler()
        data = np.random.rand(20, 4).astype(np.float32)
        scaler.fit(data)
        scaled = scaler.transform(data)
        unscaled = scaler.inverse_transform(scaled)
        cfg = scaler.get_config()
        assert isinstance(cfg, dict)
        scaler.set_config(cfg)

class TestExceptionTypes:
    @pytest.mark.parametrize("name", [
        "HeteroSymNNError", "BackendNotAvailableError", "MethodMigrationError",
        "CompilationWarning", "JITError", "ShapeMismatchError", "LayerConfigurationError",
        "WrapperError", "LoadingError", "SavingError", "RuntimeStateError",
        "HeteroSymNNValueError", "DeviceSelectionError", "ComputationalMethodValueError"
    ])
    def test_exception_subclass(self, name):
        exc_cls = getattr(exceptions, name)
        assert issubclass(exc_cls, BaseException)

class TestLayersLowLevel:
    def test_linear_layer_interface(self):
        init = initializers.HeNormal()
        layer_config = ([("relu", {})], init)
        layer = layers.LinearLayer(num_inputs=4, layer_configuration=layer_config, batch_size=8)
        x = np.random.rand(8, 4).astype(np.float32)
        x_backend = layer._CALCULATION_MANAGER.array(x.T)
        out = layer.forward(x_backend)
        err = layer._CALCULATION_MANAGER.array(np.random.rand(*out.shape).astype(np.float32))
        layer.backward(err)
        params = layer.get_parameters()
        layer.set_parameters(params)
        cfg = layer.get_config()
        assert isinstance(cfg, dict)