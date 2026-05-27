# HeteroSymNN: The Heterogeneous Activation Engine
[![PyPI version](https://badge.fury.io/py/HeteroSymNN.svg)](https://pypi.org/project/HeteroSymNN/)
[![Python versions](https://img.shields.io/pypi/pyversions/HeteroSymNN.svg)](https://pypi.org/project/HeteroSymNN/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/heterosymnn/badge/?version=latest)](https://heterosymnn.readthedocs.io/en/latest/?badge=latest)

A symbolic JIT-Compiled Deep Learning framework for Heterogeneous Neural Networks.

## What is it?

HeteroSymNN is a specialized Deep Learning engine built Control Systems and Scientific Machine Learning.

Unlike standard frameworks (PyTorch, TensorFlow) that optimize for homogeneous layers, HeteroSymNN uses a Symbolic JIT Compiler to generate fused kernels at runtime. This allows every single neuron in a layer to have a distinct, custom mathematical activation function (e.g., ```sin(num)```, ```tanh(num)```, ```alpha * num + beta```) with minimal computational overhead.

Good framework for Neuroevolution (`In planing stages`)  or Scientific ML projects but not exclusive to them.

## Table of Contents

- [Installation](#installation)  
- [Quickstart](#quickstart)  
- [Key Features](#key-features)  
- [How It Works](#how-it-works)  
- [Examples](#examples)  
- [Documentation](https://heterosymnn.readthedocs.io/en/latest/index.html)  
- [License](#license)  

## Installation

HeteroSymNN is designed to be lightweight and portable. By default, it runs on CPU using NumPy with no heavy dependencies.

### Standard Installation (CPU)

```sh
pip install HeteroSymNN
```

### GPU Installation (CUDA)

To enable the high-performance CUDA backend using CuPy:
```sh
pip install HeteroSymNN[gpu]
```
> **Note:** Requires an NVIDIA GPU and compatible CUDA drivers.

## Quickstart

HeteroSymNN follows a Scikit-Learn style API. Here is how to create a "Cocktail Layer" that mixes periodic and linear features.

```python
from HeteroSymNN.API import Wrapper
from HeteroSymNN.Core.Nets import LinearNet

# Each layer gets its own symbolic activation function
model = LinearNet(
    nodes_structure=[10, 25, 25, 1],
    activation_config=["sin(num)", "Max(0, num)", ("tanh(num)*a", {"a": 2.0})]
)

agent = Wrapper(model, work_type="reg")
agent.fit(X_train, y_train, epochs=100, batch_size=64)
```

Need per-neuron control? Use `HeteroLinearNet` to assign a unique function to every single neuron:

```python
from HeteroSymNN.Core.Nets import HeteroLinearNet

# 3 neurons, each with a different math function, repeated 4x = 12 neurons
hidden_activations = [
    ("sin(num)", {}),                       # Periodic
    ("Max(0, num)", {}),                    # ReLU
    ("exp(num * beta)", {"beta": -0.5})     # Parameterized Exponential
] * 4

model = HeteroLinearNet(
    nodes_structure=[2, 12, 1],
    detailed_activations=[hidden_activations, [("num", {})]]
)
```

## Key Features

### Per-Neuron Heterogeneity
Assign a unique mathematical activation to every single neuron. The JIT compiler fuses all distinct instructions into a single kernel launch.

### Zero-Recompile Tuning
Symbolic constants (like `alpha`, `beta`) are treated as mutable kernel arguments. Update them dynamically without triggering C++/CUDA recompilation:

```python
# Format: {layer_index: [(neuron_index, constant_name, new_value), ...]}
model.change_constants({0: [(2, "beta", -0.9)]})
```

### Save & Load Models
Serialize your entire model (architecture, weights, optimizer state, and scaler config) into a portable `.symnn` archive:

```python
# Save
agent.save_model("my_model.symnn")

# Load
loaded_agent = Wrapper.load_model("my_model.symnn")
```

### Hyperparameter Grid Search
Automatically clone and test different configurations to find the optimal model:

```python
from HeteroSymNN.API import GridSearchManager

param_grid = {'learning_rate': [0.01, 0.05, 0.1], 'batch_size': [32, 64]}

gs = GridSearchManager(agent, param_grid, validation_split=0.2)
gs.load_data(X_train, y_train)
best_agent, best_params, results = gs.execute_search(metric_to_optimize="R2")
```

## How It Works

HeteroSymNN acts as a Differentiable Compiler:

```mermaid
flowchart TD
    A[String Input: 'alpha * sin(num)'] --> B(1. Parse: SymPy AST)
    B --> C(2. Derive: Symbolic Derivative)
    C --> D(3. Compile: JIT C++/CUDA)
    D --> E(4. Fuse: Unified Hardware Kernel)
    E --> F[BackendArray Execution]
```

1. **Parse**: Accepts mathematical strings (```"alpha * sin(num)"```) and parses them into abstract syntax trees using SymPy.

2. **Derive**: Automatically calculates the exact symbolic derivative for backpropagation.

3. **Compile**: Generates C++ or CUDA code at runtime, creating a ```switch``` statement that routes each neuron to its specific math instruction.

4. **Fuse**: Fuses the memory access into a single kernel launch, avoiding the "kernel launch overhead".

The framework automatically selects the best available backend: **GPU_CUDA** → **CPU_PYTHON** (NumPy).

## Examples

The [`Examples/`](Examples/) directory contains ready-to-run scripts demonstrating different features:

| Script | What it demonstrates |
|--------|---------------------|
| `Simple_Example.py` | Basic MLP training, real-time constant tuning, and model save/load |
| `Heterogeneous_Activation_Layers_Example.py` | Per-neuron heterogeneity with 5 different activations in a single layer |
| `Grid_Search_Optimization_Example.py` | Hyperparameter grid search for XOR classification |
| `Hardware_and_Symbolic_JIT_Example.py` | GPU diagnostics, VRAM tracking, and custom symbolic JIT compilation |

## Documentation
The full documentation is hosted on [Read the Docs](https://heterosymnn.readthedocs.io/en/latest/index.html).

## License
Code released under the [MIT License](https://github.com/Dilosch03/HeteroSymNN/blob/main/LICENSE). 

[Go to Top](#table-of-contents)