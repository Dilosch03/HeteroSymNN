Type Definitions
================

Sometimes HeteroSymNN uses type definitions in type hints. This section explains what they represent.

.. module:: HeteroSymNN.types

.. data:: NodeConfig
   :annotation: = tuple[str, dict[str, float]]

   Structure for the configuration of a single node's function activation.

   * **Format:** ``("activation_expression", {"constant_name": value})``
   * **Description:** The string must be a valid mathematical expression parsable by **SymPy** or highly used activation function.
   * **Example:** ``("relu", {})`` or ``("a/(1+e**-x)", {"a": 0.8})``

.. data:: FlexibleNodeConfig
   :annotation: = str | NodeConfig

   Flexible type for node configuration. It allows using a simple string if no custom constants are needed.

   * **String:** ``"sin(x)"`` (Constants dictionary defaults to empty).
   * **Tuple:** ``("tanh(z)*a", {"a": 2.0})``.

.. data:: LayerValues
   :annotation: = tuple[list[float], list[list[float]], list[list[float]]]

   A tuple containing the raw parameter matrices for a layer. This is typically the output of an `Initializer`.

   **Structure & Dimensions:**

   1. **Biases:** ``list[float]`` of shape ``[n_nodes]``.
   2. **Weights:** ``list[list[float]]`` of shape ``[n_nodes][n_inputs]``.
   3. **Connection Mask:** ``list[list[float]]`` of shape ``[n_nodes][n_inputs]``.

   * **Format:** ``(biases, weights, connection_mask)``

.. data:: LayerConstructionConfig
   :annotation: = tuple[list[NodeConfig], LayerValues]

   The master configuration data required to initialize a ``Layer`` class.
   It combines the detailed activation configuration for every node with the initial parameter values.