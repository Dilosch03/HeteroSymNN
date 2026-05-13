# HeteroSymNN: The Heterogeneous Activation Engine
[![PyPI version](https://badge.fury.io/py/HeteroSymNN.svg)](https://pypi.org/project/HeteroSymNN/)
[![Python versions](https://img.shields.io/pypi/pyversions/HeteroSymNN.svg)](https://pypi.org/project/HeteroSymNN/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/heterosymnn/badge/?version=latest)](https://heterosymnn.readthedocs.io/en/latest/?badge=latest)

A symbolic JIT-Compiled Deep Learning framework for Heterogeneous Neural Networks.

## What is it?

HeteroSymNN is a specialized Deep Learning engine built for Neuroevolution, Control Systems, and Scientific Machine Learning.

Unlike standard frameworks (PyTorch, TensorFlow) that optimize for homogeneous layers, HeteroNN uses a Symbolic JIT Compiler to generate fused kernels at runtime. This allows every single neuron in a layer to have a distinct, custom mathematical activation function (e.g., ```sin(num)```, ```tanh(num)```, ```alpha * num + beta```) with zero computational overhead.

Good framework for Neuroevolution (NEAT) or Scientific ML projects but not exclusive to them.

## Table of Contents

-[Installation](#installation)  
-[Quickstart](#quickstart)  
-[How It Works](#how-it-works)  
-[Documentation](https://heterosymnn.readthedocs.io/en/latest/index.html)  
-[License](#license)  

## Installation

HeteroSymNN is designed to be lightweight and portable. By default, it runs in pure Python mode with no heavy dependencies.

### Standard Installation (CPU)

For basic usage (Python mode) or if you have a C++ compiler installed (for CPU JIT acceleration):
```sh
pip install HeteroSymNN
```
*Note: For CPU JIT acceleration, a C++ compiler (```g++```, ```clang```, or ```cl.exe```) is optional but highly recommended.*

### GPU Installation (CUDA)

To enable the high-performance CUDA backend using CuPy:
```sh
pip install HeteroSymNN[gpu]
```

## Quickstart

HeteroNN follows a Scikit-Learn style API. Here is how to create a "Cocktail Layer" that mixes periodic and linear features.

```python
from HeteroSymNN.API import Wrapper
from HeteroSymNN.Core.Nets import Dense

model = Dense(
    nodes_structure=[10, 25, 25, 1],
    activation_config=["sin(num)", "num", ("tanh(num)*a", {"a": 2.0})]
)

agent = Wrapper(model, work_type="reg")
agent.fit(X_train, y_train, epochs=100, batch_size=64)
```

## How It Works

HeteroNN acts as a Differentiable Compiler:

```mermaid
flowchart TD
    A[String Input: 'alpha * sin(num)'] --> B(1. Parse: SymPy AST)
    B --> C(2. Derive: Symbolic Derivative)
    C --> D(3. Compile: JIT C++/CUDA)
    D --> E(4. Fuse: Unified Hardware Kernel)
    E --> F[BackendArray Execution]
```

1. Parse: It accepts mathematical strings (```"alpha * sin(num)"```) and parses it for compilation using SymPy.

2. Derive: It automatically calculates the symbolic derivative for backpropagation.

3. Compile: It generates C++ or CUDA code at runtime, creating a ```switch``` statement that routes each neuron to its specific math instruction.

4. Fuse: It fuses the memory access into a single kernel launch, avoiding the "kernel launch overhead".

5. Dynamic Constants (Zero-Recompile Tuning): You can pass a dictionary of constants (e.g., ```{'alpha': 0.5}```) to the JIT compiler. These are treated as kernel arguments, allowing you to update hyperparameters like ```alpha``` in real-time without triggering a slow recompilation.

## Documentation
The documentation is hosted in [Read the Docs](https://heterosymnn.readthedocs.io/en/latest/index.html).

## License
Code released under the [MIT License](https://github.com/Dilosch03/HeteroSymNN/blob/main/LICENSE). 

[Go to Top](#table-of-contents)