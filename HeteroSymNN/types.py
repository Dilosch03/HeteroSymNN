from typing import TypeAlias,Union


NodeConfig: TypeAlias = tuple[str, dict[str, float]]  # Ejemplo: ("relu", {"alpha": 0.01})
FlexibleNodeConfig: TypeAlias = Union[str, NodeConfig]
LayerValues: TypeAlias = tuple[list[float], list[list[float]], list[list[float]]] # LayerValues contiene: (Biases, Weights, ConnectionMask)
LayerConstructionConfig: TypeAlias = tuple[list[NodeConfig], LayerValues] 
