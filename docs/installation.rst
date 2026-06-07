.. _installation:

Installation
============

HeteroSymNN is designed to be lightweight and portable. Its architecture allows it to gracefully degrade from High-Performance GPU execution down to pure Python based on your system's available hardware and software.

Standard Installation (CPU)
---------------------------

The standard installation relies on CPU execution. 

.. code-block:: bash

    pip install HeteroSymNN

.. note::
    **JIT Acceleration Context:** Currently the ``CPU_JIT`` mode is not available due to optimization problems that will be resolved in the future.

GPU Installation (CUDA)
-----------------------

To enable the highest-performance ``GPU_CUDA`` backend, install HeteroSymNN with the ``gpu`` flag. This leverages CuPy to compile and execute symbolic kernels directly on the GPU.

.. code-block:: bash

    pip install HeteroSymNN[gpu]

.. warning::
    This requires an NVIDIA GPU and compatible CUDA drivers pre-installed on your system for CuPy to work.

Installation from Source
------------------------

If you wish to contribute to the framework or run the bleeding-edge version, you can install directly from the repository:

.. code-block:: bash

    git clone https://github.com/Dilosch03/HeteroSymNN.git
    cd HeteroSymNN
    pip install -e .

Verifying Your Backend
----------------------

Once installed, you can quickly verify which hardware backend HeteroSymNN has selected and all the available options by checking the settings:

.. code-block:: bash
    HeteroSymNN --hardware

Expected Output:

.. code-block:: text
    Default Compute Method: CPU_PYTHON
    Available Compute Methods: CPU_PYTHON, GPU_CUDA
    Detected GPUs: 1

.. code-block:: python

    from HeteroSymNN import settings

    print(settings.available_methods)
    print(settings.default_compute_method)