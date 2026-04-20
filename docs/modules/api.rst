.. _high-level-api:

High-Level API
==============

While the Core modules handle the intense mathematics, JIT compilation, and hardware memory routing, the **API** module acts as the translation boundary for the user. 

It wraps the complex internal physics engine in a clean, standardized interface inspired by Scikit-Learn. This allows researchers to quickly instantiate, train, save, and manage networks without needing to manually write complex training loops, batching logic, or data-scaling pipelines.

Why Read This Section?
----------------------

You should consult this section to understand how to interact with your compiled networks during the actual training and deployment phases:

* **Training & Inference:** To learn how the :class:`~HeteroSymNN.API.Wrapper` class automates the forward/backward passes, epochs, batching, and model serialization.
* **Data Pipeline:** To see how the :class:`~HeteroSymNN.API.DataTransformer` safely scales, normalizes, and prepares raw data arrays before they hit the JIT-compiled engine.
* **Framework Customization:** To understand how the internal :class:`~HeteroSymNN.API.Registry` maps simple string names (like ``"relu"`` or ``"adam"``) to specific Python classes or symbolic equations.

API Components
--------------

.. toctree::
   :maxdepth: 1
   :caption: End-User Interfaces:

   apis/wrappers
   apis/data_transformers

.. toctree::
   :maxdepth: 1
   :caption: Internal Management:
   
   apis/registries