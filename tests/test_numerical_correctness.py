"""
Numerical correctness tests.

The rest of the suite checks that things *run*. These tests check that the
numbers are *right*: activation values, symbolic derivatives, per-neuron
routing, loss gradients and full-network gradients are all compared against
independent references (NumPy formulas or central finite differences).

Everything runs in float32 (the only precision the framework currently
supports), so finite differences use a fairly large step and tolerances.
The bugs these tests target are off by whole factors (e.g. 1/batch_size),
not by rounding noise.
"""
import sys
import os
import math
import pytest
import numpy as np
import warnings

# Ensure HeteroSymNN is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HeteroSymNN.config import settings
from HeteroSymNN.Core import losses, optimizers
from HeteroSymNN.Core.Nets import HeteroLinearNet, LinearNet
from HeteroSymNN.API import Wrapper

BACKENDS = [m for m in ("CPU_PYTHON", "GPU_CUDA") if m in settings.available_methods]

_erf = np.vectorize(math.erf)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# (config, NumPy reference). Every alias in codegen.COMMON_FORMULAS that is an
# activation should appear here, plus a few free-form formulas with constants.
ACTIVATION_REFERENCES = [
    (("relu", {}), lambda x: np.maximum(0.0, x)),
    (("leaky_relu", {"alpha": 0.1}), lambda x: np.where(x < 0, 0.1 * x, x)),
    (("sigmoid", {}), _sigmoid),
    (("swish", {"beta": 1.5}), lambda x: x * _sigmoid(1.5 * x)),
    (("silu", {}), lambda x: x * _sigmoid(x)),
    (("softplus", {}), lambda x: np.log1p(np.exp(x))),
    (("mish", {}), lambda x: x * np.tanh(np.log1p(np.exp(x)))),
    (("gelu", {}), lambda x: 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))),
    (("linear", {}), lambda x: x),
    (("sin(num)", {}), np.sin),
    (("tanh(num)*a + b*num", {"a": 1.5, "b": 0.2}), lambda x: 1.5 * np.tanh(x) + 0.2 * x),
    (("exp(-num**2 * c)", {"c": 0.5}), lambda x: np.exp(-0.5 * x**2)),
    (("Max(0, num)", {}), lambda x: np.maximum(0.0, x)),
]
ACTIVATION_IDS = [cfg[0] for cfg, _ in ACTIVATION_REFERENCES]

# Points kept away from the kinks of relu / leaky_relu so central differences are valid.
_GRID = np.concatenate([np.linspace(-3.0, -0.1, 12), np.linspace(0.1, 3.0, 12)]).astype(np.float32)


@pytest.fixture(autouse=True)
def _quiet_settings():
    original_level = settings.warning_level
    original_method = settings.default_compute_method
    settings.set_warning_level("ignore")
    yield
    settings.set_warning_level(original_level)
    settings.set_default_compute_method(original_method)


def _identity_net(node_configs, backend, batch_size):
    """Single-layer network with W = I and b = 0, so outputs are exactly f_j(x_j)."""
    n = len(node_configs)
    net = HeteroLinearNet(num_inputs=n, detailed_activations=[list(node_configs)], batch_size=batch_size)
    params = net.get_parameters()
    params["layer_0"]["weights"] = np.eye(n, dtype=np.float32)
    params["layer_0"]["biases"] = np.zeros((1, n), dtype=np.float32)
    net.set_parameters(params)
    return _use_backend(net, backend)


def _use_backend(net, backend):
    net.set_backend(backend)
    if backend == "GPU_CUDA":
        net.to("device")
    return net


def _to_numpy(net, value):
    return np.asarray(net.asnumpy(value), dtype=np.float64)


# =============================================================================
# 1. Activation forward values and symbolic derivatives
# =============================================================================
@pytest.mark.parametrize("backend", BACKENDS)
class TestActivationValues:

    @pytest.mark.parametrize("config,reference", ACTIVATION_REFERENCES, ids=ACTIVATION_IDS)
    def test_forward_matches_reference(self, backend, config, reference):
        net = _identity_net([config], backend, batch_size=_GRID.size)
        out = net.predict(_GRID.reshape(-1, 1))[:, 0]
        np.testing.assert_allclose(out, reference(_GRID.astype(np.float64)), rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("config,reference", ACTIVATION_REFERENCES, ids=ACTIVATION_IDS)
    def test_derivative_matches_finite_difference(self, backend, config, reference):
        net = _identity_net([config], backend, batch_size=_GRID.size)
        x = net.cast_arrays(_GRID.reshape(1, -1))
        net.forward(x)
        net.backward(net.cast_arrays(np.ones((1, _GRID.size), dtype=np.float32)))
        analytic = _to_numpy(net, net.input_gradient)[0]

        h = 1e-3
        grid64 = _GRID.astype(np.float64)
        numeric = (reference(grid64 + h) - reference(grid64 - h)) / (2 * h)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-4)


# =============================================================================
# 2. Per-neuron routing inside one heterogeneous layer
# =============================================================================
@pytest.mark.parametrize("backend", BACKENDS)
class TestHeterogeneousRouting:

    def test_every_neuron_uses_its_own_function(self, backend):
        configs = [cfg for cfg, _ in ACTIVATION_REFERENCES]
        refs = [ref for _, ref in ACTIVATION_REFERENCES]
        n, b = len(configs), 7
        x = np.random.default_rng(0).uniform(-2.5, 2.5, size=(b, n)).astype(np.float32)

        net = _identity_net(configs, backend, batch_size=b)
        out = net.predict(x)

        for j, ref in enumerate(refs):
            np.testing.assert_allclose(
                out[:, j], ref(x[:, j].astype(np.float64)), rtol=1e-4, atol=1e-5,
                err_msg=f"neuron {j} ({configs[j][0]}) was routed to the wrong function or constants",
            )

    def test_heterogeneous_derivatives_are_routed(self, backend):
        configs = [cfg for cfg, _ in ACTIVATION_REFERENCES]
        refs = [ref for _, ref in ACTIVATION_REFERENCES]
        n = len(configs)
        # Same grid for every neuron, one grid point per sample.
        x = np.tile(_GRID.reshape(-1, 1), (1, n))
        net = _identity_net(configs, backend, batch_size=x.shape[0])

        net.forward(net.cast_arrays(x.T))
        net.backward(net.cast_arrays(np.ones((n, x.shape[0]), dtype=np.float32)))
        analytic = _to_numpy(net, net.input_gradient)

        h, grid64 = 1e-3, _GRID.astype(np.float64)
        for j, ref in enumerate(refs):
            numeric = (ref(grid64 + h) - ref(grid64 - h)) / (2 * h)
            np.testing.assert_allclose(analytic[j], numeric, rtol=1e-3, atol=1e-4,
                                       err_msg=f"derivative of neuron {j} ({configs[j][0]})")

    def test_mixed_constant_counts_use_correct_offsets(self, backend):
        # 0, 1, 2 and 1 constants per neuron: exercises the offset table.
        configs = [
            ("sin(num)", {}),
            ("num * a", {"a": 3.0}),
            ("num * a + b", {"a": -1.0, "b": 5.0}),
            ("num + a", {"a": 0.25}),
        ]
        x = np.array([[0.5, 1.0, -2.0, 4.0], [1.5, -0.3, 0.0, -1.0]], dtype=np.float32)
        expected = np.stack([np.sin(x[:, 0]), 3.0 * x[:, 1], -x[:, 2] + 5.0, x[:, 3] + 0.25], axis=1)

        net = _identity_net(configs, backend, batch_size=2)
        np.testing.assert_allclose(net.predict(x), expected, rtol=1e-5, atol=1e-6)

    def test_change_constants_only_affects_target_neuron(self, backend):
        configs = [("num * a", {"a": 2.0}), ("num * a", {"a": 2.0}), ("num * a + b", {"a": 1.0, "b": 0.0})]
        x = np.array([[1.0, 1.0, 1.0], [2.0, -2.0, 3.0]], dtype=np.float32)
        net = _identity_net(configs, backend, batch_size=2)
        before = net.predict(x)

        net.change_constants({0: [(1, "a", -4.0), (2, "b", 10.0)]})
        after = net.predict(x)

        np.testing.assert_allclose(after[:, 0], before[:, 0])       # untouched neuron
        np.testing.assert_allclose(after[:, 1], -4.0 * x[:, 1])     # same formula as neuron 0, new constant
        np.testing.assert_allclose(after[:, 2], x[:, 2] + 10.0)


# =============================================================================
# 3. Loss contract: forward is a scalar mean, backward is its exact gradient
# =============================================================================
LOSS_FACTORIES = {
    "MSELoss": lambda: losses.MSELoss(),
    "MAELoss": lambda: losses.MAELoss(),
    "HuberLoss": lambda: losses.HuberLoss(),
    "BinaryCrossEntropy": lambda: losses.BinaryCrossEntropy(),
    "FlexibleLoss_custom": lambda: losses.FlexibleLoss("(y_pred - y_true)**4 + c*y_pred", constants={"c": 0.3}),
}

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("loss_name", list(LOSS_FACTORIES))
def test_loss_backward_is_gradient_of_forward(backend, loss_name):
    """
    The training loop needs `forward` to return the mean loss (a scalar) and `backward` to return
    d(forward)/d(y_pred) with the same reduction. If either side changes its reduction the
    gradients silently scale with the batch size.
    """
    loss = LOSS_FACTORIES[loss_name]()
    loss._change_COMPUTATIONAL_METHOD(backend)
    rng = np.random.default_rng(1)
    y_pred_np = rng.uniform(0.15, 0.85, size=(2, 6)).astype(np.float32)
    y_true_np = rng.integers(0, 2, size=(2, 6)).astype(np.float32)
    y_pred, y_true = loss.be.asarray(y_pred_np), loss.be.asarray(y_true_np)

    value = loss.forward(y_pred, y_true)
    assert np.size(loss.asnumpy(value)) == 1, "Loss.forward must reduce to a single scalar (the mean loss)."

    grad = np.asarray(loss.asnumpy(loss.backward(y_pred, y_true)), dtype=np.float64)
    assert grad.shape == y_pred_np.shape

    h = 1e-2
    numeric = np.zeros_like(grad)
    for idx in np.ndindex(y_pred_np.shape):
        plus, minus = y_pred_np.copy(), y_pred_np.copy()
        plus[idx] += h
        minus[idx] -= h
        f_plus = float(loss.asnumpy(loss.forward(loss.be.asarray(plus), y_true)))
        f_minus = float(loss.asnumpy(loss.forward(loss.be.asarray(minus), y_true)))
        numeric[idx] = (f_plus - f_minus) / (float(plus[idx]) - float(minus[idx]))

    np.testing.assert_allclose(grad, numeric, rtol=2e-2, atol=2e-4)


# =============================================================================
# 4. Full-network gradients (backprop through every layer)
# =============================================================================
def _network_gradient_check(net, x_np, y_np, checks_per_tensor=4, seed=0):
    x, y = net.cast_arrays(x_np, y_np)

    def loss_value():
        return float(net.asnumpy(net.loss_function.forward(net.forward(x), y)))

    y_pred = net.forward(x)
    net.backward(net.loss_function.backward(y_pred, y))
    analytic = [{k: np.array(net.asnumpy(v), dtype=np.float64) for k, v in layer.working_gradients.items()}
                for layer in net.layers]

    rng = np.random.default_rng(seed)
    failures = []
    for li, layer in enumerate(net.layers):
        for name, grad in analytic[li].items():
            param = layer.working_parameters[name]
            flat_indices = rng.choice(grad.size, size=min(checks_per_tensor, grad.size), replace=False)
            for flat in flat_indices:
                idx = np.unravel_index(flat, grad.shape)
                original = float(param[idx])
                h = 1e-2 * max(1.0, abs(original))
                param[idx] = original + h
                hi, f_plus = float(param[idx]), loss_value()
                param[idx] = original - h
                lo, f_minus = float(param[idx]), loss_value()
                param[idx] = original
                numeric = (f_plus - f_minus) / (hi - lo)
                if not np.isclose(grad[idx], numeric, rtol=3e-2, atol=1e-4):
                    failures.append(f"layer {li} {name}{idx}: analytic={grad[idx]:+.6g} numeric={numeric:+.6g} "
                                    f"(ratio {numeric / grad[idx] if grad[idx] else float('nan'):.3g})")
    return failures


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_linear_net_gradients_match_finite_differences(backend, batch_size):
    np.random.seed(0)
    net = LinearNet(nodes_structure=[3, 5, 4, 2],
                    activation_config=["tanh(num)", ("sin(num*a)", {"a": 0.7}), "linear"],
                    batch_size=batch_size)
    _use_backend(net, backend)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, batch_size)).astype(np.float32)
    y = rng.normal(size=(2, batch_size)).astype(np.float32)

    failures = _network_gradient_check(net, x, y)
    assert not failures, "Backprop gradients disagree with finite differences:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [1, 16])
def test_hetero_net_gradients_match_finite_differences(backend, batch_size):
    np.random.seed(0)
    hidden = [("sin(num)", {}), ("tanh(num)*a", {"a": 1.3}), ("softplus", {}), ("swish", {"beta": 0.8})] * 2
    net = HeteroLinearNet(num_inputs=3, detailed_activations=[hidden, [("sigmoid", {})]],
                          loss_function=losses.BinaryCrossEntropy(), batch_size=batch_size)
    _use_backend(net, backend)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, batch_size)).astype(np.float32)
    y = rng.integers(0, 2, size=(1, batch_size)).astype(np.float32)

    failures = _network_gradient_check(net, x, y)
    assert not failures, "Backprop gradients disagree with finite differences:\n  " + "\n  ".join(failures)


# =============================================================================
# 5. Training behaviour that depends on correct gradient scaling
# =============================================================================
def _sgd_step(params, batch_size, x_single, y_single, backend, lr=0.1):
    net = LinearNet(nodes_structure=[3, 4, 1], activation_config=["tanh(num)", "linear"],
                    optimizer=optimizers.SgdOptimizer(learning_rate=lr), batch_size=batch_size)
    _use_backend(net, backend)
    net.set_parameters(params)
    x, y = net.cast_arrays(np.repeat(x_single, batch_size, axis=1), np.repeat(y_single, batch_size, axis=1))
    y_pred = net.forward(x)
    net.backward(net.loss_function.backward(y_pred, y))
    net.update_params()
    return net.get_parameters()


@pytest.mark.parametrize("backend", BACKENDS)
def test_sgd_step_is_invariant_to_duplicated_samples(backend):
    """
    With a mean-reduced loss, one SGD step on B copies of the same sample must equal one step on
    that sample alone. Adam hides gradient-scale bugs (it normalises magnitudes), SGD does not.
    """
    np.random.seed(0)
    template = LinearNet(nodes_structure=[3, 4, 1], activation_config=["tanh(num)", "linear"], batch_size=1)
    params = template.get_parameters()
    x_single = np.array([[0.3], [-1.2], [0.8]], dtype=np.float32)
    y_single = np.array([[0.5]], dtype=np.float32)

    after_1 = _sgd_step(params, 1, x_single, y_single, backend)
    after_32 = _sgd_step(params, 32, x_single, y_single, backend)

    for layer_key in after_1:
        for name in ("weights", "biases"):
            np.testing.assert_allclose(after_32[layer_key][name], after_1[layer_key][name], rtol=1e-4, atol=1e-6,
                                       err_msg=f"{layer_key}.{name} depends on how many times the sample was duplicated")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sgd_recovers_linear_function_with_minibatches(backend):
    """y = 2x - 1 should be learned almost exactly by a single linear neuron trained with mini-batch SGD."""
    settings.set_default_compute_method(backend)
    rng = np.random.default_rng(4)
    X = rng.uniform(-1.0, 1.0, size=(256, 1)).astype(np.float32)
    y = (2.0 * X - 1.0).astype(np.float32)

    np.random.seed(0)
    net = LinearNet(nodes_structure=[1, 1], activation_config=["linear"],
                    optimizer=optimizers.SgdOptimizer(learning_rate=0.2), batch_size=32)
    agent = Wrapper(net, work_type="reg")
    history = agent.fit(X, y, epochs=60, batch_size=32)

    params = net.get_parameters()["layer_0"]
    assert float(params["weights"].ravel()[0]) == pytest.approx(2.0, abs=0.05)
    assert float(params["biases"].ravel()[0]) == pytest.approx(-1.0, abs=0.05)
    assert history[-1] < 1e-3
