"""
Documentation consistency tests.

README.md, LLMs_coding.rst and LLMs_dev.rst are what users and coding assistants read
first, so they must match the code. These tests fail when a documented class, method,
parameter, alias, exception, setting, file or code example no longer exists or no
longer runs.

LLMs_coding.rst API sections use a fixed format that these tests parse:

    *ClassName*  or  *ClassA, ClassB*
    - `method(param: type = default, ...) -> return`

Parameter NAMES (and their order) are checked against ``inspect.signature``.
Type annotations in the docs are free text.
"""
import sys
import os
import re
import glob
import inspect
import importlib
import warnings
import pytest
import numpy as np

# Ensure HeteroSymNN is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HeteroSymNN import exceptions
from HeteroSymNN.config import settings
from HeteroSymNN.JIT import codegen
from HeteroSymNN.Core import Nets, losses, optimizers, initializers, layers
from HeteroSymNN.API import wrappers, data_transformers, registries

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGE_DIR = os.path.join(ROOT, "HeteroSymNN")
README = os.path.join(ROOT, "README.md")
LLMS_CODING = os.path.join(ROOT, "LLMs_coding.rst")
LLMS_DEV = os.path.join(ROOT, "LLMs_dev.rst")

CLASS_NAMESPACE = {}
for _module in (Nets, losses, optimizers, initializers, layers, wrappers, data_transformers):
    CLASS_NAMESPACE.update({name: obj for name, obj in vars(_module).items() if inspect.isclass(obj)})


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _section(text, tag):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    assert match, f"Section <{tag}> not found"
    return match.group(1)


def _split_top_level(text):
    parts, depth, current = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return parts


def _documented_param_names(arg_text):
    names = []
    for part in _split_top_level(arg_text):
        name = re.split(r"[:=]", part, maxsplit=1)[0].strip().lstrip("*")
        if name and name not in ("self", "cls", "/"):
            names.append(name)
    return names


def _real_param_names(func):
    return [name for name in inspect.signature(func).parameters if name not in ("self", "cls")]


def _parse_api_entries(text):
    """Yields (class_name, method_name, documented_param_names, line) from `*Class*` / `- \\`method(...)\\`` blocks."""
    header = re.compile(r"^\*([A-Za-z_][\w, ]*)\*\s*$")
    method = re.compile(r"^- `(\w+)\((.*?)\)(?:\s*->[^`]*)?`")
    current = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("<"):
            current = []
        elif header.match(line):
            current = [name.strip() for name in header.match(line).group(1).split(",")]
        elif current and method.match(line):
            name, args = method.match(line).groups()
            for cls_name in current:
                yield cls_name, name, _documented_param_names(args), line


def _python_blocks(text):
    return re.findall(r"```python\n(.*?)```", text, re.S)


API_ENTRIES = list(_parse_api_entries(_read(LLMS_CODING)))


# =============================================================================
# LLMs_coding.rst
# =============================================================================
class TestLLMsCodingAPI:

    def test_api_sections_were_parsed(self):
        documented_classes = {cls_name for cls_name, *_ in API_ENTRIES}
        assert {"BaseNetwork", "LinearNet", "Wrapper", "GridSearchManager", "FlexibleLoss", "AdamOptimizer"} <= documented_classes

    @pytest.mark.parametrize("cls_name,method_name,doc_params,line", API_ENTRIES,
                             ids=[f"{c}.{m}" for c, m, _, _ in API_ENTRIES])
    def test_documented_signature_matches_code(self, cls_name, method_name, doc_params, line):
        assert cls_name in CLASS_NAMESPACE, f"Documented class {cls_name} does not exist"
        cls = CLASS_NAMESPACE[cls_name]
        assert hasattr(cls, method_name), f"{cls_name}.{method_name} is documented but does not exist"
        real_params = _real_param_names(getattr(cls, method_name))
        assert doc_params == real_params, (
            f"{cls_name}.{method_name}: documented parameters {doc_params} != real parameters {real_params}\n  doc line: {line}"
        )

    def test_documented_aliases_exist(self):
        section = _section(_read(LLMS_CODING), "symbolic_formulas")
        activations = re.search(r"Accepted String Activations:(.*)", section).group(1)
        loss_line = re.search(r"Accepted String Losses:(.*)", section).group(1)
        documented = set(re.findall(r'`"(\w+)"`', activations)) | set(re.findall(r'`"(\w+)"`', loss_line))
        assert documented == set(codegen.COMMON_FORMULAS), (
            f"Aliases documented but missing: {documented - set(codegen.COMMON_FORMULAS)}; "
            f"aliases in codegen but undocumented: {set(codegen.COMMON_FORMULAS) - documented}"
        )

    def test_documented_functions_match_validator_allowlist(self):
        from HeteroSymNN.JIT import validator
        section = _section(_read(LLMS_CODING), "symbolic_formulas")
        functions_line = re.search(r"Supported functions:(.*?)(?:, plus|\n)", section).group(1)
        documented = set(re.findall(r"`(\w+)`", functions_line))
        assert documented == set(validator.MATH_FUNCTIONS), (
            f"Documented but not allowed: {documented - validator.MATH_FUNCTIONS}; "
            f"allowed but undocumented: {validator.MATH_FUNCTIONS - documented}"
        )

    def test_documented_exceptions_exist(self):
        section = _section(_read(LLMS_CODING), "settings_and_exceptions")
        names = set(re.findall(r"`(\w+(?:Error|Warning))`", section))
        assert names, "No exceptions found in <settings_and_exceptions>"
        missing = sorted(n for n in names if not hasattr(exceptions, n))
        assert not missing, f"Documented exceptions that do not exist: {missing}"

    def test_documented_settings_exist(self):
        section = _section(_read(LLMS_CODING), "settings_and_exceptions")
        attributes = re.search(r"attributes:(.*)", section).group(1)
        names = re.findall(r"`(\w+)` \(", attributes)
        methods = re.findall(r"`(\w+)\(", re.search(r"Methods:(.*)", section).group(1))
        missing = [n for n in names + methods if not hasattr(settings, n)]
        assert not missing, f"Documented settings that do not exist: {missing}"


# =============================================================================
# LLMs_dev.rst
# =============================================================================
class TestLLMsDev:

    def test_directory_structure_matches_package(self):
        section = _section(_read(LLMS_DEV), "directory_structure")
        documented = set(re.findall(r"(\w+\.py)", section))
        real = {os.path.basename(p) for p in glob.glob(os.path.join(PACKAGE_DIR, "**", "*.py"), recursive=True)
                if os.path.basename(p) != "__init__.py" or os.path.dirname(p) == PACKAGE_DIR}
        assert documented - real == set(), f"Documented files that do not exist: {sorted(documented - real)}"
        assert real - documented == set(), f"Package files missing from the documented tree: {sorted(real - documented)}"

    def test_documented_folders_exist(self):
        section = _section(_read(LLMS_DEV), "directory_structure")
        for folder in re.findall(r"/(\w+):", section):
            assert glob.glob(os.path.join(PACKAGE_DIR, "**", folder), recursive=True), f"Folder /{folder} does not exist"

    @pytest.mark.parametrize("tag,base_name", [("networks", "BaseNetwork"), ("api_wrapper", "Wrapper")])
    def test_documented_method_lists_exist(self, tag, base_name):
        section = _section(_read(LLMS_DEV), tag)
        methods_line = re.search(r"# Methods: (.*)", section).group(1)
        cls = CLASS_NAMESPACE[base_name]
        missing = [m.strip() for m in methods_line.split(",") if not hasattr(cls, m.strip())]
        assert not missing, f"{base_name} methods documented but missing: {missing}"

    def test_documented_registry_methods_exist(self):
        section = _section(_read(LLMS_DEV), "registries")
        for name in set(re.findall(r"registry\.(add_\w+)", section)):
            assert hasattr(registries.registry, name), f"registry.{name} does not exist"

    def test_executable_contract_file_exists(self):
        section = _section(_read(LLMS_DEV), "executable_contract")
        path = re.search(r"(Examples/\w+\.py)", section).group(1)
        assert os.path.isfile(os.path.join(ROOT, path))


# =============================================================================
# Imports, file references and runnable examples
# =============================================================================
DOC_FILES = [README, LLMS_CODING, LLMS_DEV] + glob.glob(os.path.join(ROOT, "docs", "**", "*.rst"), recursive=True)

def _doc_imports():
    for path in DOC_FILES:
        for lineno, line in enumerate(_read(path).splitlines(), 1):
            match = re.search(r"from (HeteroSymNN[\w.]*) import ([\w, ]+)", line)
            if match:
                yield os.path.relpath(path, ROOT), lineno, match.group(1), match.group(2)

DOC_IMPORTS = list(_doc_imports())

@pytest.mark.parametrize("path,lineno,module,names", DOC_IMPORTS, ids=[f"{p}:{n}" for p, n, _, _ in DOC_IMPORTS])
def test_documented_imports_resolve(path, lineno, module, names):
    mod = importlib.import_module(module)
    for name in (n.strip() for n in names.split(",") if n.strip()):
        assert hasattr(mod, name) or importlib.util.find_spec(f"{module}.{name}"), \
            f"{path}:{lineno}: `from {module} import {name}` fails"


def test_readme_example_scripts_exist():
    table_rows = re.findall(r"^\| `([\w.]+\.py)` \|", _read(README), re.M)
    assert table_rows, "No example scripts found in the README table"
    for script in table_rows:
        assert os.path.isfile(os.path.join(ROOT, "Examples", script)), f"README lists missing example {script}"


@pytest.fixture
def example_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original_level = settings.warning_level
    settings.set_warning_level("ignore")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(96, 10)).astype(np.float32)
    y = (0.5 * X[:, :1] - 0.2 * X[:, 1:2]).astype(np.float32)
    y_class = (X[:, :1] > 0).astype(np.float32)
    yield {"X_train": X, "y_train": y, "X_test": X[:8], "y_test": y[:8],
           "y_train_class": y_class, "y_test_class": y_class[:8], "np": np}
    settings.set_warning_level(original_level)


@pytest.mark.parametrize("doc_path", [README, LLMS_CODING], ids=["README.md", "LLMs_coding.rst"])
def test_python_examples_run(doc_path, example_data):
    """Runs every ```python block of the document in order, in one shared namespace."""
    blocks = _python_blocks(_read(doc_path))
    assert blocks
    namespace = dict(example_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, block in enumerate(blocks):
            try:
                exec(compile(block, f"{os.path.basename(doc_path)}[block {i}]", "exec"), namespace)
            except Exception as e:
                pytest.fail(f"{os.path.basename(doc_path)} python block {i} failed: {type(e).__name__}: {e}\n{block}")
