Command-Line Interface (CLI)
============================

HeteroSymNN includes a command-line interface that allows you to manage framework settings, test symbolic parsing, inspect trained models, and more, directly from your terminal.

To use the CLI, simply invoke the module via python:

.. code-block:: bash

    python -m HeteroSymNN <command> [options]

Available Commands
------------------

hardware
~~~~~~~~

The ``hardware`` command queries the system to detect the available hardware and the current default compute method assigned by HeteroSymNN.

.. code-block:: bash

    python -m HeteroSymNN hardware

**Output Example:**

.. code-block:: text

    Default Compute Method: GPU_CUDA
    Available Compute Methods: ['CPU_PYTHON', 'GPU_CUDA']
    Default used CPU threads: 16
    Detected GPUs: 1
    Detected CPUs: 16

defaults
~~~~~~~~

The ``defaults`` command allows you to view and modify global framework settings natively, such as clearing the JIT cache or changing the default computing method.

**View current defaults:**

.. code-block:: bash

    python -m HeteroSymNN defaults

**Set a new property:**

Use the ``--set`` flag along with the ``--value`` flag to change global defaults. These changes are saved permanently to the user's ``settings.json`` configuration file and will persist across all future executions.

.. code-block:: bash

    # Change the default compute method to Python's NumPy
    python -m HeteroSymNN defaults --set compute --value CPU_PYTHON
    
    # Change the number of CPU threads allocated for execution
    python -m HeteroSymNN defaults --set threads --value 8
    
    # Enable or disable the kernel cache
    python -m HeteroSymNN defaults --set use-cache --value false

**Available properties:**
- ``compute``: ``GPU_CUDA``, ``CPU_JIT``, ``CPU_PYTHON``
- ``cpu-cache``: ``<path/to/directory>``
- ``warnings``: ``error``, ``ignore``, ``always``, ``default``, ``module``, ``once``
- ``threads``: ``<integer>``
- ``use-cache``: ``true``, ``false``

**Cache management:**

.. code-block:: bash

    # Clear the CPU JIT cache
    python -m HeteroSymNN defaults --clear-cpu-cache

    # Display the directory where the GPU CuPy cache is stored
    python -m HeteroSymNN defaults --show-gpu-cache

parse
~~~~~

The ``parse`` command is a powerful utility for testing mathematical strings before you add them to your model topology. It passes your string directly through HeteroSymNN's Symbolic JIT Compiler to verify it evaluates correctly and properly detects custom constants.

.. code-block:: bash

    python -m HeteroSymNN parse "sin(num)" "alpha * max(0, num)" --show-parsed --show-derivative

**Options:**
- ``--show-parsed``: Displays the internal functional expression exactly as interpreted by SymPy.
- ``--show-derivative``: Displays the exact symbolic derivative computed by the engine.

inspect
~~~~~~~

The ``inspect`` command provides a clean summary of a trained ``.symnn`` archive without needing to actually initialize the model or compile its components in a Python script. It's incredibly useful for peering into a model's topology, dynamic constants, and architecture versions.

.. code-block:: bash

    python -m HeteroSymNN inspect my_model.symnn

**Output Example:**

.. code-block:: text

    ==================================================
     METADATA 
    ==================================================
    Name:        My Custom Model
    Description: Trained overnight on the sensor dataset
    Saved At:    2026-06-07T15:18:34.123456
    Framework:   v0.3.0b7
    Task Type:   REG

    ==================================================
     TOPOLOGY 
    ==================================================
    Architecture Class: BaseNetwork
    Layer 0 [LinearLayer]: 10 -> 25  | Activations: 25x 'sin(num)'
    Layer 1 [LinearLayer]: 25 -> 25  | Activations: 15x 'relu', 10x 'tanh(num)'
    Layer 2 [LinearLayer]: 25 -> 1   | Activations: 1x 'num'

    ==================================================
     DYNAMIC CONSTANTS 
    ==================================================
    Layer 0: alfa: 1.5, beta: 0.1
    Layer 1: gamma: -0.5

    ==================================================
     PRE-PROCESSING 
    ==================================================
    Input Transformer:  MinMaxScaler
    Output Transformer: None
    ==================================================

clone
~~~~~

The ``clone`` command allows you to copy the exact architecture (including transformation configurations and metadata) from a previously saved ``.symnn`` file into a brand new, untrained model. 

.. code-block:: bash

    python -m HeteroSymNN clone original_model.symnn new_model.symnn --name "Retrained Architecture"

**Options:**
- ``--name``: Update the new model's metadata name.
- ``--description``: Update the new model's description.
- ``--custom-scripts``: Paths to Python files containing any custom layer or activation classes that must be registered before the CLI can rebuild the architecture.
