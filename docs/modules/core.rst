.. _core-components:

Core Components
===============

The **Core** module is the engine room of HeteroSymNN. While the API module handles the user-friendly Scikit-Learn style training loops, the Core module houses the raw physical memory managers, network topologies, and symbolic mathematical rules.

.. toctree::
   :maxdepth: 1
   :caption: Network Architectures:

   networks

.. toctree::
   :maxdepth: 1
   :caption: Optimization & Physics:

   rest_core/initializers
   rest_core/losses
   rest_core/optimizers

.. toctree:: 
   :maxdepth: 1
   :caption: Hardware & Memory Routing:

   layers