from __future__ import annotations
from typing import Optional,Literal,Union

from ...types import NodeConfig,LayerValues,FlexibleNodeConfig
from ..layers import LinearLayer
from .base_classes import BaseNetwork
from .. import losses as lossC, optimizers as OptiC, initializers as InitC
from ...exceptions import NetworkStructureError
from ...error_handlers import clean_traceback

__all__ = ["HeteroLinearNet", "LinearNet", "MLP"]

class HeteroLinearNet(BaseNetwork):
    """
        Base class for creating a linear_net neural network with customizable activation functions per neuron, and training parameters.
        
        Parameters
        ----------
        nodes_structure : list[int]
            List with the number of nodes per layer including input and output layers.
        detailed_activations : list[list[:type:`~HeteroSymNN.types.NodeConfig`]]
            List of lists containing the activation configuration for each node in each layer.
        initial_values : Optional[list[:type:`~HeteroSymNN.types.LayerValues`]], optional
            Optional list of initial values for each layer. If not provided, weights and biases will be initialized using the specified initializer., by default None
        initializer : List[:class:`~HeteroSymNN.Core.Nets.initializers.Initializer`], optional
            List of initializer to use for weights and biases if initial_values is not provided. Most pass a list of instance of :class:`~HeteroSymNN.Core.Nets.initializers.Initializer` and the default uses :class:`~HeteroSymNN.Core.Nets.initializers.HeNormal`, value by default is None.
        learning_rate : float, optional
            Learning rate for the network. In the case that a custom optimizer is provided with its own learning rate this value will be overwritten., by default 0.001
        batch_size : int, optional
            Batch size to use during training, by default 32. If set to -1, it uses the full dataset size for batch training.
        loss_function : :class:`~HeteroSymNN.Core.Nets.losses.Loss`, optional
            Loss function to use during training. Must be an instance of :obj:`~HeteroSymNN.Core.Nets.losses.Loss`. If not provided, :class:`~HeteroSymNN.Core.Nets.losses.MSELoss` will be used., value by default is None.
        optimizer : Optional[:class:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`], optional
            Optimizer to use for updating the network parameters. Must be an instance of :class:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`. If not provided, :class:`~HeteroSymNN.Core.Nets.optimizers.AdamOptimizer` will be used.,value by default is None.
        num_training_iter : int, optional
            Number of Epochs to use during training, by default 1000

        Attributes
        ----------
        num_training_epochs : int, read-write
            Number of training iterations (epochs) for the network.
        batch_size : int, read-write
            Batch size to use during training.
        history_losses : list[float], read-only
            List of loss values recorded at each epoch during training.
        num_completed_train_iterations : int, read-only
            Number of completed training steps.
        num_completed_epochs : int, read-only
            Number of completed training epochs.
        
        Examples
        --------
        >>> from HeteroSymNN.Core.Nets.LinearNet import HeteroLinearNet
        >>> CNN = HeteroLinearNet(
        ...     nodes_structure=[3, 5, 2],
        ...     detailed_activations=[
        ...         [("relu", {}), ("relu", {}), ("relu", {}), ("relu", {}), ("relu", {})],
        ...         [("sigmoid", {}), ("sigmoid", {})]
        ...     ],
        ...     learning_rate=0.01,
        ...     batch_size=16
        ... )

        
        >>> from HeteroSymNN.Core.Nets.LinearNet import HeteroLinearNet
        >>> from HeteroSymNN.Core import losses as lossC, optimizers as OptiC, initializers as InitC
        >>> custom_loss = lossC.CrossEntropyLoss()
        >>> custom_optimizer = OptiC.SGDOptimizer(learning_rate=0.01)
        >>> custom_initializers = [InitC.XavierUniform(),InitC.XavierUniform()]
        >>> CNN = HeteroLinearNet(
        ...     nodes_structure=[4, 6, 3],
        ...     detailed_activations=[
        ...         [("tanh", {}), ("tanh", {}), ("tanh", {}), ("tanh", {}), ("tanh", {}), ("tanh", {})],
        ...         [("softmax", {}), ("softmax", {}), ("softmax", {})]
        ...     ],
        ...     initializer=custom_initializers,
        ...     loss_function=custom_loss,
        ...     optimizer=custom_optimizer
        ... )
        
    """
    @clean_traceback
    def __init__(self, num_inputs:int, detailed_activations:list[list[NodeConfig]], 
                 initializer: Optional[list[InitC.Initializer]]= None, learning_rate:float = 0.001, batch_size:int = 32, 
                 loss_function: Optional[lossC.Loss]= None, optimizer: Optional[OptiC.Optimizer]= None, num_training_iter:int = 1000):
        network_structure = [LinearLayer] * (len(detailed_activations))

        extra_parameters = [{}]*len(detailed_activations)
        super().__init__(num_inputs,network_structure, extra_parameters, detailed_activations, initializer, learning_rate, batch_size, loss_function, optimizer, num_training_iter)



class LinearNet(HeteroLinearNet):
    """
    Intermediate class for creating linear_net neural networks with diferent activation functions per layer.
    Child class of :obj:`~HeteroSymNN.Core.Nets.LinearNet.HeteroLinearNet`.
    
    Parameters
    ----------
    nodes_structure : list[int]
        List containing the number of nodes in each layer including input and output layers.
    activation_config : list[:type:`~HeteroSymNN.types.FlexibleNodeConfig`]
        List containing the activation configuration for each layer. Each element can be a string (activation name) or a tuple (activation name, parameters dictionary).
    initial_values : Optional[list[:type:`~HeteroSymNN.types.LayerValues`]], optional
        List of initial values for weights and biases for each layer. If not provided, weights and biases will be initialized using the specified initializer., by default None
    initializer : Optional[:class:`~HeteroSymNN.Core.Nets.initializers.Initializer` | list[:class:`~HeteroSymNN.Core.Nets.initializers.Initializer`]], optional
        Initializer to use for initializing weights and biases. If not provided, :class:`~HeteroSymNN.Core.Nets.initializers.HeNormal` will be used., by default None
    learning_rate : float, optional
        Learning rate for the network., by default 0.001
    batch_size : int, optional
        Batch size to use during training. If set to -1, it uses the full dataset size for batch training., by default 32
    loss_function : :class:`~HeteroSymNN.Core.Nets.losses.Loss`, optional
        Loss function to use for training. If not provided, :class:`~HeteroSymNN.Core.Nets.losses.MSELoss` will be used., by default None
    optimizer : :class:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`, optional
        Optimizer to use for training. If not provided, :class:`~HeteroSymNN.Core.Nets.optimizers.AdamOptimizer` will be used., by default None
    num_training_iter : int, optional
        Number of training iterations (epochs)., by default 1000

    Attributes
    ----------
    num_training_epochs : int, read-write
        Number of training iterations (epochs) for the network.
    batch_size : int, read-write
        Batch size to use during training.
    history_losses : list[float], read-only
        List of loss values recorded at each epoch during training.
    num_completed_train_iterations : int, read-only
        Number of completed training steps.
    num_completed_epochs : int, read-only
        Number of completed training epochs.

    Examples
    --------
        >>> from HeteroSymNN.Core.Nets import LinearNet
        >>> FNN = LinearNet(
        ...     nodes_structure=[3, 5, 2],
        ...     activation_config=[
        ...         "relu",
        ...         ("sigmoid", {})
        ...     ],
        ...     learning_rate=0.01,
        ...     batch_size=16
        ... )
    """
    @clean_traceback
    def __init__(self, nodes_structure: list[int], activation_config: list[FlexibleNodeConfig],initializer: Optional[Union[InitC.Initializer,list[InitC.Initializer]]] = None,
                 learning_rate: float = 0.001, batch_size: int = 32,
                 loss_function: Optional[lossC.Loss] = None, optimizer: Optional[OptiC.Optimizer] = None, num_training_iter: int = 1000):
        
        num_layers = len(nodes_structure) - 1

        if (len(nodes_structure) < 2):
            raise NetworkStructureError("nodes_structure most have at least 2 value, the number of inputs and the number of outputs.")
        
        if not isinstance(activation_config, list):
             raise ValueError(f"activation_config must be a list but received: {type(activation_config)}")
        
        if len(activation_config) != num_layers:
             raise NetworkStructureError(f"The list of the activation functions have {len(activation_config)} elements, but was set {num_layers} layers in nodes_structure.")
        
        if not isinstance(initializer, list):
            if not(initializer is None):
                initializer = [initializer] * num_layers

        detailed_activations = self._expand_to_detailed(num_layers, nodes_structure[1:], activation_config)

        super().__init__(
            num_inputs=nodes_structure[0],
            detailed_activations=detailed_activations,
            initializer=initializer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer=optimizer, 
            num_training_iter=num_training_iter
        )

    def _process_node_config(self, config_item: FlexibleNodeConfig) -> NodeConfig:
        """
        Internal method to process a flexible node configuration into a strict node configuration.
        
        Parameters
        ----------
        config_item : :type:`~HeteroSymNN.types.FlexibleNodeConfig`
            Flexible node configuration (string or tuple).
            
        Returns
        -------
        :type:`~HeteroSymNN.types.NodeConfig`
            Strict node configuration (tuple)."""
        if (isinstance(config_item, str)):
            return (config_item, {})
        elif ((isinstance(config_item, tuple)) and (len(config_item) == 2)):
            return config_item
        else:
            raise ValueError(f"Invalid activation configuration: {config_item}.'str' or 'tuple[str, dict[str, float]]' was expected.")

    def _expand_to_detailed(self, num_layers: int, nodes_per_layer: list[int], layer_configs: list[FlexibleNodeConfig]) -> list[list[NodeConfig]]:
        """
        Internal method to expand flexible layer configurations into detailed node configurations for each layer.
        
        Parameters
        ----------
        num_layers : int
            Number of layers in the network.
        nodes_per_layer : list[int]
            List containing the number of nodes in each layer.
        layer_configs : list[:type:`~HeteroSymNN.types.FlexibleNodeConfig`]
            List containing the flexible activation configuration for each layer.

        Returns
        -------
        list[list[:type:`~HeteroSymNN.types.NodeConfig`]]
            Detailed activation configuration for each node in each layer.
        """
        final_config = []
        for i in range(num_layers):
            layer_conf_raw = layer_configs[i]
            num_nodes = nodes_per_layer[i]
            
            node_conf = self._process_node_config(layer_conf_raw)
            
            final_config.append([node_conf] * num_nodes)
            
        return final_config
    

class MLP(LinearNet):
    """
    High-level class for creating linear_net neural networks with uniform activation functions across hidden layers.
    
    Parameters
    ----------
    nodes_structure : list[int]
        List containing the number of nodes in each layer including input and output layers.
    activation : :type:`~HeteroSymNN.types.FlexibleNodeConfig`, optional
        Activation function configuration for hidden layers. Can be a string (activation name) or a tuple (activation name, parameters dictionary)., by default "relu
    output_activation : :type:`~HeteroSymNN.types.FlexibleNodeConfig`, optional
        Activation function configuration for the output layer. Can be a string (activation name) or a tuple (activation name, parameters dictionary)., by default "num
    initializer : Optional[:class:`~HeteroSymNN.Core.Nets.initializers.Initializer` | list[:class:`~HeteroSymNN.Core.Nets.initializers.Initializer`]], optional
        Initializer to use for initializing weights and biases. If not provided, :class:`~HeteroSymNN.Core.Nets.initializers.HeNormal` will be used., by default None
    learning_rate : float, optional
        Learning rate for the network., by default 0.001
    batch_size : int, optional
        Batch size to use during training. If set to -1, it uses the full dataset size for batch training., by default 32
    loss_function : :obj:`~HeteroSymNN.Core.Nets.losses.Loss`, optional
        Loss function to use for training. If not provided, :obj:`~HeteroSymNN.Core.Nets.losses.MSELoss` will be used., by default None
    optimizer : :obj:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`, optional
        Optimizer to use for training. If not provided, :obj:`~HeteroSymNN.Core.Nets.optimizers.AdamOptimizer` will be used., by default None
    num_training_iter : int, optional
        Number of training iterations (epochs)., by default 1000
    
    Attributes
    ----------
    num_training_epochs : int, read-write
        Number of training iterations (epochs) for the network.
    batch_size : int, read-write
        Batch size to use during training.
    history_losses : list[float], read-only
        List of loss values recorded at each epoch during training.
    num_completed_train_iterations : int, read-only
        Number of completed training steps.
    num_completed_epochs : int, read-only
        Number of completed training epochs.

    Examples
    --------
        >>> from HeteroSymNN.Core.Nets import MLP
        >>> SNN = MLP(
        ...     nodes_structure=[4, 8, 3],
        ...     activation="tanh",
        ...     output_activation="softmax",
        ...     learning_rate=0.01,
        ...     batch_size=64
        ... )
    """
    @clean_traceback
    def __init__(self, nodes_structure: list[int], activation: FlexibleNodeConfig = "relu(x)", output_activation: FlexibleNodeConfig = "num",initializer: Optional[InitC.Initializer] = None,
                 learning_rate: float = 0.001,batch_size: int = 32, loss_function: Optional[lossC.Loss] = None, optimizer: Optional[OptiC.Optimizer] = None, num_training_iter: int = 1000):

        if len(nodes_structure) < 2:
            raise NetworkStructureError("node_structure must have at least 2 elements (input layer and output layer).")

        num_hidden_layers = len(nodes_structure) - 2 
        
        activations_list = [activation] * num_hidden_layers
        activations_list.append(output_activation)

        super().__init__(
            nodes_structure=nodes_structure,
            activation_config=activations_list,
            initializer=initializer, 
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer=optimizer,
            num_training_iter=num_training_iter
        )