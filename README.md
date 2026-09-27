# HeteroSymNN: The Heterogeneous Activation Engine
[![PyPI version](https://badge.fury.io/py/HeteroSymNN.svg)](https://pypi.org/project/HeteroSymNN/)
[![Python versions](https://img.shields.io/pypi/pyversions/HeteroSymNN.svg)](https://pypi.org/project/HeteroSymNN/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/heterosymnn/badge/?version=latest)](https://heterosymnn.readthedocs.io/en/latest/?badge=latest)

A symbolic, JIT-compiled neural network library where **every neuron can have its own activation function**, written as a plain math string.

## What is it?

You describe activations as formulas (`"sin(num)"`, `"tanh(num)*a"`, `"exp(num*beta)"`). HeteroSymNN parses them with SymPy, derives the exact gradient symbolically, and compiles all the functions of a layer into **one** CUDA kernel. Each neuron is routed to its own formula inside that kernel. Constants such as `a` or `beta` are stored per neuron and can be changed at any time **without recompiling**.

It is aimed at Scientific Machine Learning, symbolic and physics-inspired models, and neuroevolution-style experiments, where the activation functions themselves are part of what you are exploring.

**When it is a good fit**
- You want different, hand-written activation functions per neuron without writing custom kernels or autograd code.
- You change activation constants often (tuning, sensitivity studies, evolutionary search).
- You want a small dependency footprint (NumPy + SymPy, CuPy optional).

**When it is not**
- Large standard architectures (CNNs, Transformers) or maximum raw throughput: PyTorch and JAX are faster on homogeneous networks (see [Performance](#performance)).
- Anything beyond fully-connected layers today: recurrent layers, convolutions and normalization are on the [roadmap](https://heterosymnn.readthedocs.io/en/latest/roadmap.html).

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Key Features](#key-features)
- [Performance](#performance)
- [How It Works](#how-it-works)
- [Examples](#examples)
- [Project Status](#project-status)
- [Documentation](#documentation)
- [License](#license)

## Installation

HeteroSymNN runs on CPU with NumPy by default, with no heavy dependencies.

### Standard Installation (CPU)

```sh
pip install HeteroSymNN
```

### GPU Installation (CUDA)

To enable the CUDA backend (via CuPy), pick the extra that matches your CUDA version (`cuda11`, `cuda12` or `cuda13`):
```sh
pip install HeteroSymNN[cuda12]
```
> **Note:** Requires an NVIDIA GPU and compatible drivers. If CuPy cannot be loaded, HeteroSymNN falls back to the CPU with a warning.

## Quickstart

HeteroSymNN follows a Scikit-Learn style API. Data is passed as `(samples, features)` arrays.

```python
from HeteroSymNN.API import Wrapper
from HeteroSymNN.Core.Nets import LinearNet

# One activation per layer: here periodic, then ReLU, then a scaled tanh with constant a = 2.0
model = LinearNet(
    nodes_structure=[10, 25, 25, 1],
    activation_config=["sin(num)", "Max(0, num)", ("tanh(num)*a", {"a": 2.0})]
)

agent = Wrapper(model, work_type="reg")
agent.fit(X_train, y_train, epochs=100, batch_size=64)
predictions = agent.predict(X_test)
```

Need per-neuron control? `HeteroLinearNet` takes one `(formula, constants)` pair for every neuron:

```python
from HeteroSymNN.Core.Nets import HeteroLinearNet

# 3 different functions, repeated 4 times = 12 hidden neurons
hidden_activations = [
    ("sin(num)", {}),                       # Periodic
    ("Max(0, num)", {}),                    # ReLU
    ("exp(num * beta)", {"beta": -0.5})     # Parameterized exponential
] * 4

hetero_model = HeteroLinearNet(
    num_inputs=2,
    detailed_activations=[hidden_activations, [("num", {})]]
)
```

Formula rules in one line: the input variable is `num` (`x` and `z` are accepted aliases), anything called with parentheses is a function (`sin(num)`), and any other name is a constant that you must supply in the dictionary. Built-in aliases: `relu`, `sigmoid`, `silu`, `softplus`, `mish`, `gelu`, `linear`, `leaky_relu` (needs `alpha`) and `swish` (needs `beta`).

## Key Features

### Per-Neuron Heterogeneity
Assign a different mathematical activation to every neuron. The JIT compiler puts all the distinct formulas of a layer into a single kernel launch, so adding more distinct functions does not add more kernels or intermediate tensors.

### Zero-Recompile Tuning
Symbolic constants (like `alpha` and `beta`) are kernel arguments, not compiled literals. You can update them per neuron without triggering a recompilation:

```python
# Format: {layer_index: [(neuron_index, constant_name, new_value), ...]}
hetero_model.change_constants({0: [(2, "beta", -0.9)]})
```

### Save & Load Models
Save the full model (architecture, per-neuron constants, weights, optimizer state and data-transformer configuration) into a portable `.symnn` archive (JSON and NumPy, no pickle):

```python
agent.save_model("my_model.symnn")
loaded_agent = Wrapper.load_model("my_model.symnn")
```

Formulas (including those inside a loaded `.symnn` file) are checked against a math-only allowlist before SymPy parses them, so a model file can't smuggle in Python code.

### Hyperparameter Grid Search
Clone and train one model per combination of settings, then keep the best by a validation metric (`R2`, `MSE`, … for regression, `Acur`, … for classification):

```python
from HeteroSymNN.API import GridSearchManager

param_grid = {"learning_rate": [0.01, 0.05], "batch_size": [32, 64], "num_training_epochs": [50]}

gs = GridSearchManager(agent, param_grid, validation_split=0.2)
gs.load_data(X_train, y_train)
best_agent, best_params, results = gs.execute_search(metric_to_optimize="R2")
```

### Command-Line Interface
Inspect hardware, manage defaults, inspect saved models and test formulas without writing a script:

```bash
python -m HeteroSymNN hardware                                # detected GPUs, threads, default backend
python -m HeteroSymNN inspect my_model.symnn                  # topology and constants of a saved model
python -m HeteroSymNN parse "alpha * sin(num)" --show-derivative
```

See the [CLI documentation](https://heterosymnn.readthedocs.io/en/latest/cli.html) for all commands (`defaults`, `clone`, `version`, …).

## Performance

Summary of the [benchmarks](https://heterosymnn.readthedocs.io/en/latest/benchmarks.html), run on an RTX 4060 Laptop GPU (training step, forward + backward + update):

| Scenario | Result |
|----------|--------|
| 1 → 32 distinct functions in one 1,024-neuron layer | HeteroSymNN stays flat (~1.3–1.6 ms). PyTorch grows from ~1.0 to 2.4 ms, JAX from 0.3 to 1.5 ms. PyTorch is faster below ~16 functions and JAX below ~32. |
| GPU memory, 4 functions, batch 1k → 131k | 44–76% less GPU memory than a slice-and-concatenate PyTorch baseline. |
| Standard homogeneous networks | Slower: about 1.6× PyTorch and 2–6× JAX. |
| Changing a constant every batch | No recompilation (~5 ms/batch). A JAX setup that treats the constant as a compile-time value recompiles every time. |

In short: HeteroSymNN's advantages are flexibility, memory use, and cost that stays flat as layers get more heterogeneous. It is not the fastest option for plain networks.

## How It Works

HeteroSymNN acts as a differentiable compiler:

```mermaid
flowchart TD
    A[String Input: 'alpha * sin(num)'] --> B(1. Parse: SymPy)
    B --> C(2. Derive: Symbolic Derivative)
    C --> D(3. Compile: JIT CUDA / NumPy)
    D --> E(4. Fuse: One kernel per layer)
    E --> F[BackendArray Execution]
```

1. **Parse**: The formula string is parsed into a SymPy expression. Free names become per-neuron constants.
2. **Derive**: The exact symbolic derivative is computed for backpropagation.
3. **Compile**: On GPU, CUDA C is generated and compiled in memory (NVRTC via CuPy). On CPU, the expressions become vectorized NumPy functions.
4. **Fuse**: On GPU, every distinct formula in a layer becomes one case of a `switch` statement inside a single kernel. Each neuron is routed to its case, and its constants are read from a flat parameter array.

Backends: **GPU_CUDA** when CuPy and an NVIDIA GPU are available, otherwise **CPU_PYTHON** (NumPy). You can switch at runtime with `model.set_backend(...)`.

## Examples

The [`Examples/`](https://github.com/Dilosch03/HeteroSymNN/tree/main/Examples) directory contains ready-to-run scripts:

| Script | What it demonstrates |
|--------|---------------------|
| `train_boilerplate.py` | Training with the `Wrapper`: CSV loss logging, periodic checkpoints, and a safe checkpoint on Ctrl+C |
| `custom_training_loop.py` | Writing your own loop with `forward` / `backward` / `update_params`, a validation split and early stopping |
| `transfer_learning_template.py` | Loading a trained model, transferring its parameters to a new architecture, and fine-tuning |
| `custom_extensions_template.py` | Writing and registering your own Loss, Optimizer, Initializer and DataTransformer, then saving and loading them |
| `interactive_tuning_gui.py` | A Tkinter GUI that changes activation constants live with `change_constants()` |

The [`notebooks/`](https://github.com/Dilosch03/HeteroSymNN/tree/main/notebooks) directory has guided tutorials, from the quickstart to a real-world pulsar classification example.

## Project Status

HeteroSymNN is in **beta** (v0.3.x). Current limitations:

- Fully-connected layers only (`LinearNet`, `MLP`, `HeteroLinearNet`).
- `float32` only.
- The C++/OpenMP CPU backend (`CPU_JIT`) is disabled. Requesting it falls back to `CPU_PYTHON`.
- The public API may still change between minor versions.

The [roadmap](https://heterosymnn.readthedocs.io/en/latest/roadmap.html) lists what is planned next.

## Documentation
The full documentation is hosted on [Read the Docs](https://heterosymnn.readthedocs.io/en/latest/index.html).

## License
Code released under the [MIT License](https://github.com/Dilosch03/HeteroSymNN/blob/main/LICENSE).

[Go to Top](#table-of-contents)
