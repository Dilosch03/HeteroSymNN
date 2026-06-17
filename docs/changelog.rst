.. _changelog:

Changelog
=========

All notable changes to the HeteroSymNN framework will be documented in this file. 
The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_.

[0.3.0] - 2026-04-19 (The Architecture Update)
----------------------------------------------
**Added**

* **Centralized Configuration & Error Handling:**

  * Added ``config.py`` utilizing a singleton design pattern to manage global framework defaults and automatic hardware fallback logic (``GPU_CUDA`` -> ``CPU_JIT`` -> ``CPU_PYTHON``).
  * Added ``exceptions.py`` to provide custom, framework-specific exceptions (e.g., :exec:`~HeteroSymNN.exceptions.BackendNotAvailableError`, :exec:`~HeteroSymNN.exceptions.CompilationWarning`) for safer try/except blocks.

* **New API Subsystems:**

  * Added ``API/wrappers.py`` implementing a Scikit-Learn style interface and remade the :class:`~HeteroSymNN.API.wrappers.GridSearchManager` class.
  * Added ``API/data_transformers.py`` to handle data preprocessing and scaling before routing to the neural networks.
  * Added ``API/registries.py`` to manage the dynamic auto-discovery of framework components.

* **Network & Layer Modularity:**

  * Added ``Core/Nets/base_classes.py`` to provide strict abstract base classes for network definitions.
  * Added ``Core/Nets/linear_net.py`` for specialized linear_net network implementations.
  * Added ``Core/Nets/evo.py`` laying the structural foundation for the upcoming Evolutionary Networks (EvoNets) pipeline.

* **JIT Pipeline:**

  * Added ``JIT/codegen.py`` to handle the generation of C++/CUDA code, upgrading the previous template-based approach.

* **AI & Documentation Integrations:**

  * Added ``LLMs_coding.rst`` and ``LLMs_dev.rst`` (LLM Context files) to provide a strict dependency and architectural map for AI code assistants interacting with the framework.
  * Extended the Documentation, by adding more detailed explanations of the framework's architecture and general framework context.

* **Legacy Support & Testing:**

  * Added legacy constructors (``Core/Nets/legacy_nets.py``, ``Core/legacy_core_objects.py``) for backward compatibility with previous network and layer implementations.
  * Added robust testing suites (``tests/test_capabilities.py``, ``tests/test_cli.py``, ``tests/test_jit_exceptions.py``) covering CLI operations, JIT exceptions, and framework capabilities.

* **Network Features:**

  * Added a property in the base network classes to retrieve the calculated errors with respect to the input of the network.

* **New CLI Tool:**

  * Added ``heterosymnn.cli`` for standardized command-line operations.
  * Added the ``hardware`` command for hardware detection and compatibility reporting.
  * Added the ``defaults`` command for managing framework defaults and cache, including support for permanent settings.
  * Added the ``parse`` command for testing symbolic math strings through the JIT compiler.
  * Improved CLI function parsing for better robustness.

**Changed / Refactored**

* **API & Wrappers:**

  * Optimized the ``regression_test_accuracy`` wrapper method for better performance.
  * Refactored network classes to eliminate obsolete methods, attributes, and constructor parameters, enforcing a single source of truth and aligning with industry conventions.

* **Name Changes:**

  * Changed ``SimpleNN`` class name to :customref:`MLP <mlp-network>`.
  * Changed ``FlaxibleNN`` class name to :customref:`LinearNet <dense-network>`.
  * Changed ``ConfigurableNN`` class name to :customref:`HeteroLinearNet <heterodense-network>`.
  * Changed ``Layer`` class name to :customref:`LinearLayer <linear-layer>`.

* **Kernel Allocation Optimization:**

  * Standard Optimizers (Adam, SGD) now use Static Singleton Allocation for CuPy kernels, allowing tools like the new Tuner to spawn thousands of wrapper clones without triggering recompilations.
  * :customref:`SymbolicJITCompiler <jit>` and :class:`~HeteroSymNN.Core.losses.FlexibleLoss` now utilize Instance-Level Allocation to safely handle mutating mathematical strings.

* **State Safety & Encapsulation:**

  * Direct attribute mutation on layers is now deprecated. Layer parameters must be updated using the unified :meth:`~HeteroSymNN.Core.layers.BaseLayer.set_parameters` method to guarantee hardware-routing safeguards are triggered.
  * Internal boolean states (e.g., ``_fitted`` in DataTransformers) are now strictly guarded by public read-only properties.

* **Hardware Device Routing:**

  * The framework now enforces unified device memory via a global ``to()`` call, preventing silent PCIe bottlenecks caused by mixing CPU and GPU layers mid-pipeline.

* **Directory Restructuring:**

  * Moved ``layers.py`` from ``Core/Nets/`` up to the main ``Core/`` directory to separate individual layer logic from full network assemblies.
  * ``docs/`` folder structure overhauled, splitting monolithic files into ``docs/modules/api``, ``docs/modules/core``, ``docs/modules/jit``, etc.

**Removed**

* Currently the ``CPU_JIT`` method has been deactivated due to performance issues.
* Removed ``Core/Nets/neural_nets.py`` (Logic has been modularized and distributed into ``base_classes.py``, ``linear_net.py``, and ``evo.py``).
* Removed ``JIT/templates.py`` (Replaced by the more robust and flexible string formatting pipeline in ``codegen.py``).
* Removed legacy pickle-based config loading in favor of standard JSON serialization for ``.symnn`` wrapper creation.
* Removed ``learning_mode`` network parameter.
  
[0.2.0] - Alpha Releases
------------------------
* Initial implementation of the Symbolic JIT Compiler.
* Introduction of the ``ConfigurableNN`` node-level customization builder.
* CuPy and NVIDIA NVRTC integration.

[0.1.0] - Proof of Concept
--------------------------
* Framework inception and basic SymPy string parsing logic.