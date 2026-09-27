"""
Formula-string security tests.

Formula strings are parsed with SymPy's ``parse_expr``, which uses ``eval`` internally.
Formula strings also arrive from ``.symnn`` archives, so without an allow-list, loading
a model file from someone else can execute arbitrary Python.

Each payload below tries to create a marker file. The tests assert both that the
framework refuses the formula *and* that the marker was never written, so they catch
"raised an error, but only after running the code" as well.

The last class checks the other direction: legitimate formulas (everything in the docs
and in ``codegen.COMMON_FORMULAS``) must keep working once the allow-list is in place.
"""
import sys
import os
import json
import zipfile
import pytest
from unittest.mock import patch

# Ensure HeteroSymNN is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HeteroSymNN.config import settings
from HeteroSymNN.exceptions import FormulaParsingError, HeteroSymNNError
from HeteroSymNN.JIT import codegen
from HeteroSymNN.JIT.compiler import SymbolicJITCompiler
from HeteroSymNN.Core.losses import FlexibleLoss
from HeteroSymNN.Core.Nets import LinearNet, BaseNetwork
from HeteroSymNN.cli import main


# Each template receives the marker path and must create that file if Python is evaluated.
PAYLOAD_TEMPLATES = {
    "builtin_open": "num + 0*len(open(r'{marker}', 'w').name)",
    "dunder_import": "num * (__import__('pathlib').Path(r'{marker}').touch() or 1)",
    "lambda_call": "num * ((lambda: open(r'{marker}', 'w'))() and 1)",
    "loss_builtin_open": "(y_pred - y_true)**2 + 0*len(open(r'{marker}', 'w').name)",
}
ACTIVATION_PAYLOADS = ["builtin_open", "dunder_import", "lambda_call"]

# Constructs that have no business in a math formula, even if they have no side effect.
FORBIDDEN_CONSTRUCTS = {
    "attribute_access": "num.__class__",
    "dunder_name": "__builtins__ * num",
    "subscript": "[num, 1][0]",
    "string_literal": "num * len('abc')",
    "keyword_argument": "Max(num, 0, evaluate=False)",
    "comprehension": "sum([num for _ in range(3)])",
}


@pytest.fixture(autouse=True)
def _no_kernel_cache():
    original_cache = settings.use_kernel_cache
    original_level = settings.warning_level
    settings.use_kernel_cache = False
    settings.set_warning_level("ignore")
    yield
    settings.use_kernel_cache = original_cache
    settings.set_warning_level(original_level)


def _payload(name, tmp_path):
    marker = tmp_path / f"{name}.marker"
    return PAYLOAD_TEMPLATES[name].format(marker=marker.as_posix()), marker


class TestPayloadsAreNeverExecuted:

    @pytest.mark.parametrize("name", ACTIVATION_PAYLOADS)
    def test_compiler_rejects_payload(self, name, tmp_path):
        formula, marker = _payload(name, tmp_path)
        with pytest.raises(FormulaParsingError):
            SymbolicJITCompiler([(formula, {})], "CPU_PYTHON", 0)
        assert not marker.exists(), f"Formula was evaluated as Python: {formula}"

    @pytest.mark.parametrize("name", ACTIVATION_PAYLOADS)
    def test_network_constructor_rejects_payload(self, name, tmp_path):
        formula, marker = _payload(name, tmp_path)
        with pytest.raises(HeteroSymNNError):
            LinearNet(nodes_structure=[2, 3, 1], activation_config=[formula, "linear"])
        assert not marker.exists(), f"Formula was evaluated as Python: {formula}"

    def test_flexible_loss_rejects_payload(self, tmp_path):
        formula, marker = _payload("loss_builtin_open", tmp_path)
        with pytest.raises(HeteroSymNNError):
            FlexibleLoss(formula)
        assert not marker.exists(), f"Loss expression was evaluated as Python: {formula}"

    def test_cli_parse_does_not_execute_payload(self, tmp_path, capsys):
        formula, marker = _payload("builtin_open", tmp_path)
        with patch("sys.argv", ["heterosymnn", "parse", formula]):
            try:
                main()
            except SystemExit:
                pass
        assert not marker.exists(), "`heterosymnn parse` evaluated the formula as Python"

    @pytest.mark.parametrize("name", ACTIVATION_PAYLOADS)
    def test_loading_crafted_archive_does_not_execute_payload(self, name, tmp_path):
        """A .symnn file is data. Loading one from an untrusted source must never run code."""
        model = LinearNet(nodes_structure=[2, 3, 1], activation_config=["tanh(num)", "linear"])
        clean_path = model.save_model(str(tmp_path / "clean"))

        formula, marker = _payload(name, tmp_path)
        crafted_path = tmp_path / "crafted.symnn"
        with zipfile.ZipFile(clean_path) as src, zipfile.ZipFile(crafted_path, "w") as dst:
            for item in src.namelist():
                data = src.read(item)
                if item == "config.json":
                    config = json.loads(data)
                    layer_0 = config["architecture"]["layer_configs"]["layer_0"]
                    layer_0["layer_node_configs"] = [[formula, {}] for _ in layer_0["layer_node_configs"]]
                    data = json.dumps(config).encode()
                dst.writestr(item, data)

        with pytest.raises(HeteroSymNNError):
            BaseNetwork.load_model(str(crafted_path))
        assert not marker.exists(), "Loading a crafted .symnn archive executed embedded Python"


class TestForbiddenConstructs:

    @pytest.mark.parametrize("formula", list(FORBIDDEN_CONSTRUCTS.values()), ids=list(FORBIDDEN_CONSTRUCTS))
    def test_non_math_syntax_is_rejected(self, formula):
        with pytest.raises(FormulaParsingError):
            SymbolicJITCompiler([(formula, {})], "CPU_PYTHON", 0)


class TestResourceLimits:
    """Formulas that are valid math but built to exhaust CPU or memory while SymPy processes them."""

    @pytest.mark.parametrize("formula", [
        "num + 9**9**9",                       # constant power tower: exact evaluation never finishes
        "num + 2**100000",                     # huge constant exponent
        "num * " + "+".join(["num"] * 400),    # too many nodes
        "sin(" * 40 + "num" + ")" * 40,        # too deeply nested
        "num + " + "1" * 2500,                 # too long
    ], ids=["power_tower", "huge_exponent", "too_many_nodes", "too_deep", "too_long"])
    def test_resource_exhausting_formulas_are_rejected(self, formula):
        with pytest.raises(FormulaParsingError):
            SymbolicJITCompiler([(formula, {})], "CPU_PYTHON", 0)

    def test_constant_name_cannot_expand_into_many_symbols(self):
        # sympy.symbols("x:100000000") would create 100 million symbols.
        with pytest.raises(FormulaParsingError):
            SymbolicJITCompiler([("num * x", {"x:100000000": 1.0})], "CPU_PYTHON", 0)

    @pytest.mark.parametrize("constants", [{"a b": 1.0}, {"__class__": 1.0}, {1: 2.0}, {"a": "1.0"}, {"a": [1.0]}],
                             ids=["space", "dunder", "non_string", "string_value", "list_value"])
    def test_invalid_constants_are_rejected(self, constants):
        with pytest.raises(FormulaParsingError):
            SymbolicJITCompiler([("num", constants)], "CPU_PYTHON", 0)

    def test_unicode_lookalike_names_cannot_bypass_the_allowlist(self, tmp_path):
        # Python normalizes 'ｏｐｅｎ' (full-width) to 'open' when evaluating.
        marker = tmp_path / "unicode.marker"
        formula = f"num + 0*len(ｏｐｅｎ(r'{marker.as_posix()}', 'w').name)"
        with pytest.raises(FormulaParsingError):
            SymbolicJITCompiler([(formula, {})], "CPU_PYTHON", 0)
        assert not marker.exists()


class TestLegitimateFormulasStillWork:
    """Guards against an allow-list that is too strict."""

    ACTIVATIONS = [
        ("relu", {}), ("sigmoid", {}), ("silu", {}), ("softplus", {}), ("mish", {}), ("gelu", {}), ("linear", {}),
        ("leaky_relu", {"alpha": 0.01}), ("swish", {"beta": 1.0}),
        ("sin(num)", {}), ("Max(0, num)", {}), ("Abs(num)", {}), ("num**2", {}), ("sqrt(2.0)*num", {}),
        ("exp(num * beta)", {"beta": -0.5}), ("tanh(num)*a", {"a": 2.0}), ("alpha * sin(num)", {"alpha": 1.0}),
        ("1e-7 + num", {}), ("-num / 3", {}), ("atan(num) + erf(num)", {}), ("pi * num + e", {}),
        ("Piecewise((num * alpha, num < 0), (num, True))", {"alpha": 0.2}),
        ("relu(num) * alfa", {"alfa": 2.0}), ("sin(x)", {}), ("tanh(z)", {}),
        # Nested composition of functions and aliases is part of the expressive power and must keep working.
        ("sin(tanh(softplus(num)))", {}), ("relu(gelu(num)) + sigmoid(sin(num*w))", {"w": 3.0}),
        ("Piecewise((sin(num), (num > -1) & (num < 1)), (tanh(num), True))", {}),
        ("num**-2 + 2**num + num**(1/3)", {}), ("atan2(num, 1) * cbrt(num)", {}), ("2**10 * num", {}),
    ]

    @pytest.mark.parametrize("formula,constants", ACTIVATIONS, ids=[f for f, _ in ACTIVATIONS])
    def test_activation_formula_compiles(self, formula, constants):
        SymbolicJITCompiler([(formula, constants)], "CPU_PYTHON", 0)

    @pytest.mark.parametrize("key", sorted(k for k, v in codegen.COMMON_FORMULAS.items() if "y_pred" in v))
    def test_builtin_loss_formula_compiles(self, key):
        FlexibleLoss(key)

    def test_custom_loss_with_constants_compiles(self):
        FlexibleLoss(
            "Piecewise((0.5*(y_pred-y_true)**2, Abs(y_pred-y_true)<=delta), (delta*Abs(y_pred-y_true)-0.5*delta**2, True))",
            constants={"delta": 1.0},
        )
