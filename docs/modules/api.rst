High-Level API
==============

The **API** module provides high-level wrappers to simplify the training, evaluation, and management of HeteroSymNN models.

.. automodule:: HeteroSymNN.API.wrappers
   :members:
   :undoc-members:
   :show-inheritance:

Model Wrapper
-------------

.. autoclass:: HeteroSymNN.API.wrappers.Wraper
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

   The primary interface for interacting with neural networks.
   
   It abstracts away the complexity of:
   
   * Data Normalization (Min-Max scaling).
   * Training loops.
   * Model persistence (Saving/Loading to ``.npz``).
   * Accuracy testing (Regression and Classification metrics).

   **Example Usage:**

   .. code-block:: python

      agent = Wraper(model, work_type="reg")
      agent.load_training(X_train, y_train)
      agent.run_training(num_iterations=100)
      prediction = agent.predict(X_new)

Grid Search Wrapper
-------------------

.. autoclass:: HeteroSymNN.API.wrappers.GridSearchWraper
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

   A tool for Hyperparameter Optimization.
   
   It automatically splits training data into Train/Validation sets and tests combinations of parameters.

   **Key Features:**
   
   * **Automatic Splitting:** Splits data into training and validation sets.
   * **Metric Tracking:** Tracks R2 (Regression) or Accuracy (Classification).
   * **Best Model Retention:** Automatically keeps the best performing model instance.