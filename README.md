# HeteroSymNN: The Heterogeneous Activation Engine
A symbolic JIT-Compliled Deep Learning framework for Heterogeneous Neural Networks.

## What is it?

HeteroSymNN is a specialized Deep Learning engine built for Neuroevolution, Control Systems, and Scientific Machine Learning.

Unlike standard frameworks (PyTorch, TensorFlow) that optimize for homogeneous layers, HeteroNN uses a Symbolic JIT Compiler to generate fused kernels at runtime. This allows every single neuron in a layer to have a distinct, custom mathematical activation function (e.g., ```sin(x)```, ```tanh(x)```, ```alpha * x + beta```) with zero computational overhead.

Good framework for Neuroevolution (NEAT) or Scientific ML projects but not excusive to them.

## Table of Contents

-[Installation](#installation)  
-[Quickstart](#quickstart)  
-[How It Works](#how-it-works)  
-[Documentation]()  
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

```sh
from HeteroSymNN.API.wrappers import Wraper
from HeteroSymNN.Core.Nets.neural_nets import FlexibleNN

model = FlexibleNN(
    nodes_structure=[10,25,25,1],
    activation_config = ["sin(x)","num",("tanh(z)*a",{"a":2})],
    training_mode = "mini-batch",
    batch_size = 32,
    num_treaning_iter = 200
    )

agent = Wraper(model,work_type="reg")
agent.load_training(X_train, y_train)
agent.run_training(num_iterations = 100, batch_size=64)
```

## How It Works

HeteroNN acts as a Differentiable Compiler:

1. Parse: It accepts mathematical strings (```"alpha * sin(x)"```) and parses it for compilation using SymPy.

2. Derive: It automatically calculates the symbolic derivative for backpropagation.

3. Compile: It generates C++ or CUDA code at runtime, creating a ```switch``` statement that routes each neuron to its specific math instruction.

4. Fuse: It fuses the memory access into a single kernel launch, avoiding the "kernel launch overhead".

5. Dynamic Constants (Zero-Recompile Tuning): You can pass a dictionary of constants (e.g., ```{'alpha': 0.5}```) to the JIT compiler. These are treated as kernel arguments, allowing you to update hyperparameters like ```alpha``` in real-time without triggering a slow recompilation.

## Documentation
The documentation is hosted in []().

## License
Code released under the [MIT License](). 

[Go to Top](#table-of-contents)