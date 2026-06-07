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
  * Added ``exceptions.py`` to provide custom, framework-specific exceptions (e.g., ``BackendNotAvailableError``, ``CompilationWarning``) for safer try/except blocks.

* **New API Subsystems:**

  * Added ``API/wrappers.py`` implementing a Scikit-Learn style interface and remade the ``GridSearchManager`` object.
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
  * Massive extensions to the Sphinx documentation suite, including quickstarts, module-specific API breakdowns, and robust autoclass mock object typing in ``types.py``.

**Changed / Refactored**

* **Name Changes:**

  * Changed ``SimpleNN`` class name to :class:`~HeteroSymNN.Core.Nets.linear_net.MLP`.
  * Changed ``FlaxibleNN`` class name to :class:`~HeteroSymNN.Core.Nets.linear_net.LinearNet`.
  * Changed ``ConfigurableNN`` class name to :class:`~HeteroSymNN.Core.Nets.linear_net.HeteroLinearNet`.
  * Changed ``Layer`` class name to :class:`~HeteroSymNN.Core.layers.LinearLayer`.

* **Kernel Allocation Optimization:**

  * Standard Optimizers (Adam, SGD) now use Static Singleton Allocation for CuPy kernels, allowing tools like the new Tuner to spawn thousands of wrapper clones without triggering recompilations.
  * ``SymbolicJITCompiler`` and ``FlexibleLoss`` now utilize Instance-Level Allocation to safely handle mutating mathematical strings.

* **State Safety & Encapsulation:**

  * Direct attribute mutation on layers is now deprecated. Layer parameters must be updated using the unified ``layer.set_parameters()`` dictionary API to guarantee hardware-routing safeguards are triggered.
  * Internal boolean states (e.g., ``_fitted`` in DataTransformers) are now strictly guarded by public read-only properties.

* **Hardware Device Routing:**

  * The framework now enforces unified device memory via a global ``change_device()`` call, preventing silent PCIe bottlenecks caused by mixing CPU and GPU layers mid-pipeline.

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
* Introduction of the ``HeteroLinearNet`` node-level customization builder.
* CuPy and NVIDIA NVRTC integration.

[0.1.0] - Proof of Concept
--------------------------
* Framework inception and basic SymPy string parsing logic.