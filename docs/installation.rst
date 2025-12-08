Installation
============

HeteroSymNN is designed to be lightweight and portable. By default, it runs in pure Python mode with no heavy dependencies.

Standard Installation (CPU)
---------------------------

For basic usage or if you have a C++ compiler installed (for CPU JIT acceleration):

.. code-block:: bash

    pip install HeteroSymNN

.. note::
    For CPU JIT acceleration, a C++ compiler (``g++``, ``clang``, or ``cl.exe``) is optional but highly recommended.

GPU Installation (CUDA)
-----------------------

To enable the high-performance CUDA backend using CuPy, install with the ``gpu`` extra:

.. code-block:: bash

    pip install "HeteroSymNN[gpu]"

From Source
-----------

.. code-block:: bash

    git clone https://github.com/Dilosch03/HeteroSymNN.git
    cd HeteroSymNN
    pip install -e .