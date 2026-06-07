import sys
import os
import pytest
import numpy as np

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
        model.change_constants({1: [(0, "alpha", 2.0)]})
        model.predict(X_dummy)

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
        agent.load_state(save_path)

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
        out = layer.forward(x.T)
        err = np.random.rand(*out.shape).astype(np.float32)
        layer.backward(err)
        params = layer.get_parameters()
        layer.set_parameters(params)
        cfg = layer.get_config()
        assert isinstance(cfg, dict)
