from __future__ import annotations
import os
from typing import Union, TYPE_CHECKING, TypeAlias

import numpy as np

__all__ = [
    "NodeConfig", "FlexibleNodeConfig", "LayerValues",
    "LayerConstruction", "BackendArray", "ConstantToUpdate",
]


if os.environ.get("SPHINX_BUILD") == "True":
    
    class NodeConfig:
        """NodeConfig represents the activation function and its custom constants in the function.
        
        Definition: ``tuple[str, dict[str, float]]``
        """
        pass

    class LayerValues:
        """LayerValues represents the parameters of a neural network layer, including biases, weights, and connection masks in that order.
        
        Definition: ``tuple[list[float], list[list[float]], list[list[float]]]``
        """
        pass

    class LayerConstruction:
        """LayerConstruction encapsulates the configuration needed to construct a neural network layer.
        
        Definition: ``tuple[list[NodeConfig], Initializer]``
        """
        pass

    class BackendArray:
        """BackendArray represents an array type used in the backend computations (NumPy or CuPy)."""
        pass

    class ConstantToUpdate:
        """ConstantToUpdate represents a tuple containing the node index, the constant name, and the new value.
        
        Definition: ``tuple[int, str, float]``
        """
        pass

    FlexibleNodeConfig = Union[str, NodeConfig]
    """FlexibleNodeConfig can be either a simple string representing the activation function name or a detailed :obj:`~NodeConfig` tuple."""

else:
    if (TYPE_CHECKING):
        from .Core.initializers import Initializer

    NodeConfig: TypeAlias = tuple[str, dict[str, float]]  # Example: ("relu", {"alpha": 0.01})
    """NodeConfig represents the activation function and its custom constants in the function.
        Definition: ``tuple[str, dict[str, float]]``"""
    FlexibleNodeConfig: TypeAlias = Union[str, NodeConfig]
    """FlexibleNodeConfig can be either a simple string representing the activation function name or a detailed :obj:`~NodeConfig` tuple."""
    LayerValues: TypeAlias = tuple[list[float], list[list[float]], list[list[float]]] # LayerValues contiene: (Biases, Weights, ConnectionMask)
    """LayerValues represents the parameters of a neural network layer, including biases, weights, and connection masks in that order.
        Definition: ``tuple[list[float], list[list[float]], list[list[float]]]``"""
    LayerConstruction: TypeAlias = tuple[list[NodeConfig], "Initializer"]
    """LayerConstructionConfig encapsulates the configuration needed to construct a neural network layer, including a list of :obj:`~NodeConfig` for each node and the corresponding :obj:`~LayerValues`.
         Definition: ``tuple[list[NodeConfig], Initializer]``"""
    BackendArray: TypeAlias = np.ndarray
    """BackendArray represents an array type used in the backend computations, can be either a NumPy array or a CuPy array depending on the hardware backend."""
    ConstantToUpdate: TypeAlias = tuple[int, str, float]
    """ConstantToUpdate represents a tuple containing the node index, the constant name, and the new value.
        Definition: ``tuple[int, str, float]``"""