.. _wrappers:

Wrappers & Training Interfaces
==============================

Under the hood, HeteroSymNN acts as a complex Differentiable JIT Compiler, managing hardware memory routing, symbolic derivatives, and C++ code generation. However, we make end-user experience simple and intuitive. 

The wrapper classes act as this translation boundary. They encapsulate the raw network architectures and provide standard, Scikit-Learn style interfaces (like ``fit`` and ``predict``) so you don't have to manually write complex ``for`` loops, batching chunkers, or metric trackers.

1. The Model Wrapper
--------------------

**When to Use:**
Use the ``Wrapper`` class for all day-to-day model training, evaluation, and data transformation. Whether you are running a simple classification task or a complex physics regression, this class handles the forward/backward data flow, automatic scaling (if a DataTransformer is passed), and saving and loading of model safely from/to a ``.symnn`` archive.

**Code Example:**

.. code-block:: python

    from HeteroSymNN.Core.Nets import LinearNet
    from HeteroSymNN.API import Wrapper

    # 1. Define your raw engine
    model = LinearNet(nodes_structure=[10, 25, 1], activation_config=["sin(num)", "num"])

    # 2. Wrap it for Regression ("reg") or Classification ("class")
    agent = Wrapper(model, work_type="reg")

    # 3. Train and predict instantly
    agent.fit(X_train, y_train, epochs=100)
    predictions = agent.predict(X_new)
    
    # 4. Save the compiled model and its weights
    agent.save_model("my_physics_model.symnn")

.. autoclass:: HeteroSymNN.API.wrappers.Wrapper
   :members:
   :undoc-members:

2. Grid Search Manager
----------------------

**When to Use:**
Use the ``GridSearchManager`` when you need to find the optimal hyperparameters for your network. 

Because HeteroSymNN treats mathematical constants (like ``alpha`` or ``beta``) as mutable kernel arguments, this grid searcher can test thousands of different symbolic constant values *without* triggering a slow C++/CUDA recompilation, making it drastically faster than standard framework tuning.

**Code Example:**

.. code-block:: python

    from HeteroSymNN.API import GridSearchManager
    from HeteroSymNN.Core.optimizers import AdamOptimizer, SgdOptimizer

    # 1. Define the parameter grid to test
    param_grid = {
        'learning_rate': [0.01, 0.001],
        'batch_size': [32, 64],
        'optimizer': [AdamOptimizer(), SgdOptimizer()]
    }

    # 2. Initialize the search across the wrapped model
    searcher = GridSearchManager(agent, param_grid, validation_split=0.2)
    
    # 3. Execute the search and return the best performing wrapper
    searcher.load_data(X_train, y_train)
    best_agent, best_params, results = searcher.execute_search(metric_to_optimize="R2")


.. autoclass:: HeteroSymNN.API.wrappers.GridSearchManager
   :members:
   :undoc-members: