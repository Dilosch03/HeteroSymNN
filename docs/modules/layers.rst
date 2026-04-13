Layer Classes
=============
In HeteroSymNN, layers are not just static mathematical matrices. They are the physical memory representations of your symbolic graphs. They allocate the required :data:`BackendArray` buffers (in CPU RAM or GPU VRAM) and store the execution pointers to your compiled JIT kernels.

.. toctree::
    :maxdepth: 1
    :caption: Layer classes:

    layer_classes/general_layer
    layer_classes/linear_layer