.. _exceptions:

Exceptions & Warnings
=====================

To facilitate debugging of the created neural networks the framework implements a strict, granular hierarchy of custom exceptions and warnings. This allows researchers, automated scripts, and AI coding assistants to precisely identify engine failures—such as a GPU memory migration block or a symbolic syntax error—and programmatically recover without crashing the entire Python process.

The Error Hierarchy
-------------------
All custom exceptions inherit from the base :exc:`~HeteroSymNN.exceptions.HeteroSymNNError`, and all custom warnings inherit from :exc:`~HeteroSymNN.exceptions.HeteroSymNNWarnings`. They are categorized into five main functional domains:

* **Backend Errors:** Inheriting from :exc:`~HeteroSymNN.exceptions.BackendError`. These encompass any issues related to the underlying computational hardware, physical memory management, or execution environment transitions.
* **JIT Compilation Errors:** Inheriting from :exc:`~HeteroSymNN.exceptions.JITError`. These cover all failures encountered during the dynamic translation, compilation, linking, or execution of custom mathematical kernels.
* **Configuration Errors:** Inheriting from :exc:`~HeteroSymNN.exceptions.ConfigError`. These catch structural, dimensional, or parameter-based mismatches during the initialization and setup phases of the network or its individual components.
* **API & Wrapper Errors:** Inheriting from :exc:`~HeteroSymNN.exceptions.WrapperError`. These represent high-level failures that occur during user-facing operations, such as data handling, model training routines, or state serialization and persistence.
* **Value & Selection Errors:** Inheriting from :exc:`~HeteroSymNN.exceptions.HeteroSymNNValueError`. These represent cases where an invalid option, device, or computational method has been specified, subclassing both their functional error domain and Python's native ``ValueError`` (such as :exc:`~HeteroSymNN.exceptions.DeviceSelectionError` and :exc:`~HeteroSymNN.exceptions.ComputationalMethodValueError`).

Warnings
---------
Non-fatal execution bottlenecks are communicated via custom warnings that inherit from the base :exc:`~HeteroSymNN.exceptions.HeteroSymNNWarnings` class (a subclass of Python's standard ``UserWarning``)

Debugging: Controlling Warning Behavior
---------------------------------------
When building new topologies or debugging performance drops, you might want the framework to strictly halt execution rather than silently passing a warning. Alternatively, in production, you might want to suppress warnings entirely. 

Instead of manually configuring Python's native warnings module, HeteroSymNN provides a built-in global setting to control this behavior instantly:

.. code-block:: python

    from HeteroSymNN import config

    # Escalate all HeteroSymNN warnings to fatal exceptions to catch silent bottlenecks
    config.settings.set_warning_level("error")

    # Completely silence all framework warnings for clean production logs
    config.settings.set_warning_level("ignore")
    
    # Revert to the default behavior (print warnings to console)
    config.settings.set_warning_level("default")

Code Example: Safe Fallbacks
----------------------------
By catching specific framework exceptions, you can build highly resilient training pipelines that automatically adapt to their hardware constraints.

.. code-block:: python

    from HeteroSymNN.API.wrappers import Wrapper
    from HeteroSymNN.exceptions import BackendNotAvailableError, JITError
    from HeteroSymNN import config

    try:
        # Attempt to initialize the wrapper (defaults to GPU if detected)
        agent = Wrapper(model, work_type="reg")
        agent.fit(X_train, y_train)

    except BackendNotAvailableError:
        # Catch the hardware failure and gracefully degrade the engine
        print("CUDA unavailable or out of memory. Falling back to CPU JIT...")
        config.settings.set_default_compute_method("CPU_JIT")
        agent.fit(X_train, y_train)

    except JITError as e:
        # Catch bad symbolic math strings before they crash the Python interpreter
        print(f"Mathematical Syntax Error in your custom layer activation: {e}")

API Reference
-------------

.. automodule:: HeteroSymNN.exceptions
   :members:
   :undoc-members:
   :show-inheritance: