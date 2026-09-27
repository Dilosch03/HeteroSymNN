"""
Robustness tests for behaviour that only shows up when something goes wrong
(or when state crosses a boundary): error re-raising, backend fallback, and
round-tripping per-neuron state through a save file.
"""
import sys
import os
import contextlib
import pytest
import numpy as np

# Ensure HeteroSymNN is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HeteroSymNN.config import settings
from HeteroSymNN.error_handlers import clean_traceback
from HeteroSymNN.exceptions import PerformanceWarning, ShapeMismatchError, HeteroSymNNError
from HeteroSymNN.JIT import compiler as compiler_module
from HeteroSymNN.Backend import hardware as HW
from HeteroSymNN.Core import losses, optimizers
from HeteroSymNN.Core.Nets import HeteroLinearNet, BaseNetwork, LinearNet
from HeteroSymNN.API import Wrapper


@pytest.fixture(autouse=True)
def _quiet_settings():
    original_level = settings.warning_level
    settings.set_warning_level("ignore")
    yield
    settings.set_warning_level(original_level)


# =============================================================================
# clean_traceback must never change *which* error the user sees
# =============================================================================
class TwoArgumentError(Exception):
    """Mimics exceptions whose constructor needs more than a message (UnicodeDecodeError, many library errors)."""
    def __init__(self, code, detail):
        super().__init__(f"[{code}] {detail}")
        self.code = code


class TwoArgumentFrameworkError(HeteroSymNNError):
    """A framework error with a multi-argument constructor: the kind clean_traceback does clean."""
    def __init__(self, code, detail):
        super().__init__(f"[{code}] {detail}")
        self.code = code


def _frame_names(exc_info):
    return [entry.name for entry in exc_info.traceback]


class TestCleanTraceback:

    @pytest.fixture(autouse=True)
    def _debug_off(self, monkeypatch):
        monkeypatch.setattr(settings, "debug_mode", False, raising=False)

    def test_preserves_type_and_message(self):
        @clean_traceback
        def fails():
            raise ShapeMismatchError("expected (4, 1), got (4, 2)")

        with pytest.raises(ShapeMismatchError, match=r"expected \(4, 1\), got \(4, 2\)"):
            fails()

    @pytest.mark.parametrize("error_class", [TwoArgumentError, TwoArgumentFrameworkError])
    def test_preserves_exceptions_with_multi_argument_constructors(self, error_class):
        original = error_class(42, "device lost")

        @clean_traceback
        def fails():
            raise original

        with pytest.raises(error_class) as info:
            fails()
        assert info.value is original
        assert "device lost" in str(info.value)
        assert info.value.code == 42

    def test_framework_errors_hide_internal_frames_and_chains(self):
        def internal_helper():
            try:
                {}["missing"]
            except KeyError as e:
                raise ShapeMismatchError("bad shape") from e

        @clean_traceback
        def public_method():
            internal_helper()

        with pytest.raises(ShapeMismatchError) as info:
            public_method()
        assert "internal_helper" not in _frame_names(info)
        assert info.value.__cause__ is None and info.value.__suppress_context__

    def test_unexpected_errors_keep_their_full_traceback(self):
        """Non-framework errors are bugs: the frame that raised them must stay visible."""
        def internal_helper():
            return [][0]

        @clean_traceback
        def public_method():
            internal_helper()

        with pytest.raises(IndexError) as info:
            public_method()
        assert "internal_helper" in _frame_names(info)

    def test_debug_mode_passes_errors_through_untouched(self, monkeypatch):
        monkeypatch.setattr(settings, "debug_mode", True, raising=False)
        original = TwoArgumentError(7, "raw")

        @clean_traceback
        def fails():
            raise original

        with pytest.raises(TwoArgumentError) as info:
            fails()
        assert info.value is original


# =============================================================================
# A failed CUDA compilation must leave a consistent CPU state
# =============================================================================
class _FailingCudaBackend:
    """Minimal stand-in for CuPy where NVRTC compilation always fails."""
    int32 = np.int32

    class cuda:
        @staticmethod
        def Device(_id):
            return contextlib.nullcontext()

    @staticmethod
    def RawKernel(*args, **kwargs):
        raise RuntimeError("simulated NVRTC failure")

    @staticmethod
    def array(*args, **kwargs):
        return "GPU_ARRAY_SENTINEL"


def test_failed_cuda_switch_falls_back_to_a_working_cpu_compiler(monkeypatch):
    configs = [("tanh(num)", {}), ("sin(num)", {}), ("num * a", {"a": 3.0})]
    comp = compiler_module.SymbolicJITCompiler(configs, "CPU_PYTHON", 0)

    monkeypatch.setattr(HW, "GPU_ENABLED", True)
    monkeypatch.setattr(HW, "be", _FailingCudaBackend)
    monkeypatch.setattr(compiler_module, "_validate_gpu_id", lambda _id: None)

    with pytest.warns(PerformanceWarning):
        method = comp._change_method("GPU_CUDA")

    assert method == "CPU_PYTHON"
    assert comp.calculation_method == "CPU_PYTHON"
    assert list(comp.func_ids) == list(comp.func_ids_cpu), "func_ids points at a GPU array after falling back to CPU"

    z = np.random.default_rng(0).normal(size=(3, 5)).astype(np.float32)
    a = np.empty_like(z)
    comp.forward_kernel(z, a, np.array([3.0], dtype=np.float32), np.array([0, 0, 0], dtype=np.int32), 3, 5)
    np.testing.assert_allclose(a, np.stack([np.tanh(z[0]), np.sin(z[1]), 3.0 * z[2]]), rtol=1e-5)


# =============================================================================
# Per-neuron constants survive save -> load
# =============================================================================
def test_mutated_per_neuron_constants_survive_save_and_load(tmp_path):
    hidden = [("tanh(num)*a", {"a": 1.0}), ("sin(num*w)", {"w": 2.0}), ("num", {}), ("exp(num*beta)", {"beta": -0.5})]
    model = HeteroLinearNet(num_inputs=2, detailed_activations=[hidden, [("linear", {})]], batch_size=4)
    model.change_constants({0: [(0, "a", 3.5), (3, "beta", -0.9)]})

    x = np.random.default_rng(0).normal(size=(6, 2)).astype(np.float32)
    expected = model.predict(x)

    path = model.save_model(str(tmp_path / "mutated"))
    loaded = BaseNetwork.load_model(str(path))

    np.testing.assert_allclose(loaded.predict(x), expected, rtol=1e-5, atol=1e-6)
    saved_configs = loaded.get_config()["layer_configs"]["layer_0"]["layer_node_configs"]
    assert saved_configs[0][1]["a"] == pytest.approx(3.5)
    assert saved_configs[3][1]["beta"] == pytest.approx(-0.9)


def test_save_model_return_value_can_be_passed_to_load_model(tmp_path):
    """`save_model` is annotated `-> str`; whatever it returns must be accepted by `load_model`."""
    model = LinearNet(nodes_structure=[2, 3, 1], activation_config=["tanh(num)", "linear"])
    path = model.save_model(str(tmp_path / "roundtrip"))
    assert isinstance(path, str)
    BaseNetwork.load_model(path)


# =============================================================================
# Custom losses written to the documented contract work with the Wrapper
# =============================================================================
class ContractLogCosh(losses.Loss):
    """Custom loss following the documented contract: mean scalar forward, matching gradient backward."""

    def forward(self, y_pred, y_true):
        return self.be.mean(self.be.log(self.be.cosh(y_pred - y_true)))

    def backward(self, y_pred, y_true):
        return self.be.tanh(y_pred - y_true) / y_pred.size


class SumSquaredError(losses.Loss):
    """A loss with a SUM reduction: it must define its own gradient scaling and batch combination."""

    def forward(self, y_pred, y_true):
        return self.be.sum((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true)

    def combine_batches(self, batch_losses, batch_sizes):
        return sum(batch_losses)


def _frozen_epoch_loss(loss, X, y, batch_size):
    """Runs one epoch with lr = 0 (parameters never change) and returns (reported epoch loss, full-dataset loss)."""
    np.random.seed(0)
    model = LinearNet(nodes_structure=[2, 4, 1], activation_config=["tanh(num)", "linear"], loss_function=loss,
                      optimizer=optimizers.SgdOptimizer(learning_rate=0.0), batch_size=batch_size)
    history = Wrapper(model, work_type="reg").fit(X, y, epochs=1, batch_size=batch_size)
    predictions = model.predict(X)
    full = float(model.asnumpy(loss.forward(model.cast_arrays(predictions.T), model.cast_arrays(y.T))))
    return history[0], full


@pytest.mark.parametrize("loss_factory", [losses.MSELoss, SumSquaredError], ids=["mean_reduction", "sum_reduction"])
def test_wrapper_epoch_loss_is_defined_by_the_loss(loss_factory):
    """50 samples in batches of 16 (last batch has 2): the reported epoch loss must equal the loss over the whole set."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(50, 2)).astype(np.float32)
    y = rng.normal(size=(50, 1)).astype(np.float32)
    reported, full = _frozen_epoch_loss(loss_factory(), X, y, batch_size=16)
    assert reported == pytest.approx(full, rel=1e-4)


def test_wrapper_trains_with_contract_following_custom_loss():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, size=(64, 2)).astype(np.float32)
    y = (X[:, :1] - 0.5 * X[:, 1:]).astype(np.float32)

    np.random.seed(0)
    model = LinearNet(nodes_structure=[2, 4, 1], activation_config=["tanh(num)", "linear"],
                      loss_function=ContractLogCosh(), optimizer=optimizers.AdamOptimizer(learning_rate=0.05), batch_size=16)
    history = Wrapper(model, work_type="reg").fit(X, y, epochs=30, batch_size=16)
    batch_losses = Wrapper(model, work_type="reg").fit(X, y, epochs=1, batch_size=16, return_batch_losses=True)

    assert all(np.isfinite(history))
    assert history[-1] < history[0]
    assert len(batch_losses) == 4 and all(isinstance(v, float) for v in batch_losses)
