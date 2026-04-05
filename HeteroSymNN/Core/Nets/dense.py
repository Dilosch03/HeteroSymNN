from __future__ import annotations
from typing import Optional,Literal
import itertools as iter

from ...types import NodeConfig,LayerValues,FlexibleNodeConfig,LayerConstruction,NodeConfig
from ..layers import LinearLayer
from .base_classes import BaseNetwork
from .. import losses as lossC, optimizers as OptiC, initializers as InitC
from ...exceptions import NetworkStructureError

class HeteroDense(BaseNetwork):
    """
        Base class for creating a dense neural network with customizable activation functions per neuron, and training parameters.
        
        Parameters
        ----------
        nodes_structure : list[int]
            List with the number of nodes per layer including input and output layers.
        detailed_activations : list[list[:obj:`~HeteroSymNN.types.NodeConfig`]]
            List of lists containing the activation configuration for each node in each layer.
        initial_values : Optional[list[:obj:`~HeteroSymNN.types.LayerValues`]], optional
            Optional list of initial values for each layer. If not provided, weights and biases will be initialized using the specified initializer., by default None
        initializer : Optional[:obj:`~HeteroSymNN.Core.Nets.initializers.Initializer`], optional
            Initializer to use for weights and biases if initial_values is not provided. Most pass an instance of :obj:`~HeteroSymNN.Core.Nets.initializers.Initializer` and the default uses :obj:`~HeteroSymNN.Core.Nets.initializers.HeNormal`, value by default is None.
        learning_rate : float, optional
            Learning rate for the network. In the case that a custom optimizer is provided with its own learning rate this value will be overwritten., by default 0.001
        batch_size : int, optional
            Batch size to use during training, by default 32 if training_mode is "mini-batch", 1 if "stochastic" and size of the dataset if "batch".
        training_mode : Literal["batch", "mini-batch", "stochastic"], optional
            Training mode to use during training. Options are "batch", "mini-batch", and "stochastic". By default "mini-batch".
        loss_function : :obj:`~HeteroSymNN.Core.Nets.losses.Loss`, optional
            Loss function to use during training. Must be an instance of :obj:`~HeteroSymNN.Core.Nets.losses.Loss`. If not provided, ::obj:`~HeteroSymNN.Core.Nets.losses.MSELoss` will be used., value by default is None.
        optimizer : Optional[:obj:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`], optional
            Optimizer to use for updating the network parameters. Must be an instance of :obj:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`. If not provided, :obj:`~HeteroSymNN.Core.Nets.optimizers.AdamOptimizer` will be used.,value by default is None.
        num_treaning_iter : int, optional
            Number of Epochs to use during training, by default 1000

        Attributes
        ----------
        num_treaning_iterations : int, read-write
            Number of training iterations (epochs) for the network.
        learning_mode : str, read-write
            Learning mode of the network. Currently only "Static" is supported.
        training_mode : Literal["batch", "mini-batch", "stochastic"], read-write
            Training mode to use during training. When seting it to "mini-batch" from "stochastic" or "batch" the batch size that will be used is the one stored in the attribute batch_size.
        batch_size : int, read-write
            Batch size to use during training.
        histogram_losses : list[float], read-only
            List of loss values recorded at each epoch during training.
        num_complited_train_iterations : int, read-only
            Number of completed training steps.
        num_completed_epochs : int, read-only
            Number of completed training epochs.
        
        Examples
        --------
        >>> from HeteroSymNN.Core.Nets.Dense import HeteroDense
        >>> CNN = HeteroDense(
        ...     nodes_structure=[3, 5, 2],
        ...     detailed_activations=[
        ...         [("relu", {}), ("relu", {}), ("relu", {}), ("relu", {}), ("relu", {})],
        ...         [("sigmoid", {}), ("sigmoid", {})]
        ...     ],
        ...     learning_rate=0.01,
        ...     batch_size=16,
        ...     training_mode="mini-batch"
        ... )

        
        >>> from HeteroSymNN.Core.Nets.Dense import HeteroDense
        >>> from HeteroSymNN.Core import losses as lossC, optimizers as OptiC, initializers as InitC
        >>> custom_loss = lossC.CrossEntropyLoss()
        >>> custom_optimizer = OptiC.SGDOptimizer(learning_rate=0.01)
        >>> custom_initializer = InitC.XavierUniform()
        >>> CNN = HeteroDense(
        ...     nodes_structure=[4, 6, 3],
        ...     detailed_activations=[
        ...         [("tanh", {}), ("tanh", {}), ("tanh", {}), ("tanh", {}), ("tanh", {}), ("tanh", {})],
        ...         [("softmax", {}), ("softmax", {}), ("softmax", {})]
        ...     ],
        ...     initializer=custom_initializer,
        ...     loss_function=custom_loss,
        ...     optimizer=custom_optimizer,
        ...     training_mode="batch"
        ... )
        
    """
    def __init__(self, nodes_structure:list[int], detailed_activations:list[list[NodeConfig]], initial_values: Optional[list[LayerValues]]= None, 
                 initializer: Optional[list[InitC.Initializer]]= None, learning_rate:float = 0.001, batch_size:int = 32, training_mode:str = "mini-batch", 
                 loss_function: Optional[lossC.Loss]= None, optimizer: Optional[OptiC.Optimizer]= None, num_treaning_iter:int = 1000):
        layer_types = [LinearLayer] * (len(nodes_structure)-1)
        network_structure = list(zip(nodes_structure[1:], layer_types))
        network_structure = [(nodes_structure[0], None)] + network_structure
        extra_parameters = [{}]*len(detailed_activations)
        super().__init__(network_structure, extra_parameters, detailed_activations, initial_values, initializer, learning_rate, batch_size, training_mode, loss_function, optimizer, num_treaning_iter)



class Dense(HeteroDense):
    """
    Intermediate class for creating dense neural networks with diferent activation functions per layer.
    Child class of :obj:`~HeteroSymNN.Core.Nets.Dense.HeteroDense`.
    
    Parameters
    ----------
    nodes_structure : list[int]
        List containing the number of nodes in each layer including input and output layers.
    activation_config : list[:obj:`~HeteroSymNN.types.FlexibleNodeConfig`]
        List containing the activation configuration for each layer. Each element can be a string (activation name) or a tuple (activation name, parameters dictionary).
    initial_values : Optional[list[:obj:`~HeteroSymNN.types.LayerValues`]], optional
        List of initial values for weights and biases for each layer. If not provided, weights and biases will be initialized using the specified initializer., by default None
    initializer : Optional[:obj:`~HeteroSymNN.Core.Nets.initializers.Initializer`], optional
        Initializer to use for initializing weights and biases. If not provided, :obj:`~HeteroSymNN.Core.Nets.initializers.HeNormal` will be used., by default None
    learning_rate : float, optional
        Learning rate for the network., by default 0.001
    training_mode : Literal["batch", "mini-batch", "stochastic"], optional
        Training mode to use during training. In case of "batch" or "stochastic" the batch size attribute will be ignored., by default "stochastic"
    batch_size : int, optional
        Batch size to use during training. In the case of using "stochastic" or "batch" training mode this attribute will be ignored and in training time the batch size will be set to 1 or to the full dataset size respectively., by default 32
    loss_function : :obj:`~HeteroSymNN.Core.Nets.losses.Loss`, optional
        Loss function to use for training. If not provided, :obj:`~HeteroSymNN.Core.Nets.losses.MSELoss` will be used., by default None
    optimizer : :obj:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`, optional
        Optimizer to use for training. If not provided, :obj:`~HeteroSymNN.Core.Nets.optimizers.AdamOptimizer` will be used., by default None
    num_treaning_iter : int, optional
        Number of training iterations (epochs)., by default 1000

    Attributes
    ----------
    num_treaning_iterations : int, read-write
        Number of training iterations (epochs) for the network.
    learning_mode : str, read-write
        Learning mode of the network. Currently only "Static" is supported.
    training_mode : Literal["batch", "mini-batch", "stochastic"], read-write
        Training mode to use during training. When seting it to "mini-batch" from "stochastic" or "batch" the batch size that will be used is the one stored in the attribute batch_size.
    batch_size : int, read-write
        Batch size to use during training.
    histogram_losses : list[float], read-only
        List of loss values recorded at each epoch during training.
    num_complited_train_iterations : int, read-only
        Number of completed training steps.
    num_completed_epochs : int, read-only
        Number of completed training epochs.

    Examples
    --------
        >>> from HeteroSymNN.Core.Nets import Dense
        >>> FNN = Dense(
        ...     nodes_structure=[3, 5, 2],
        ...     activation_config=[
        ...         "relu",
        ...         ("sigmoid", {})
        ...     ],
        ...     learning_rate=0.01,
        ...     batch_size=16,
        ...     training_mode="mini-batch"
        ... )
    """
    def __init__(self, nodes_structure: list[int], activation_config: list[FlexibleNodeConfig],initial_values: Optional[list[LayerValues]] = None,initializer: Optional[InitC.Initializer] = None,
                 learning_rate: float = 0.001, training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic", batch_size: int = 32,
                 loss_function: Optional[lossC.Loss] = None, optimizer: Optional[OptiC.Optimizer] = None, num_treaning_iter: int = 1000):
        
        num_layers = len(nodes_structure) - 1
        
        if not isinstance(activation_config, list):
             raise ValueError(f"activation_config must be a list but received: {type(activation_config)}")
        
        if len(activation_config) != num_layers:
             raise NetworkStructureError(f"The list of the activation functions have {len(activation_config)} elements, but was set {num_layers} layers in nodes_structure.")

        detailed_activations = self._expand_to_detailed(num_layers, nodes_structure[1:], activation_config)

        super().__init__(
            nodes_structure=nodes_structure,
            detailed_activations=detailed_activations,
            initial_values=initial_values,
            initializer=initializer,
            learning_rate=learning_rate,
            training_mode=training_mode,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer=optimizer, 
            num_treaning_iter=num_treaning_iter
        )

    def _process_node_config(self, config_item: FlexibleNodeConfig) -> NodeConfig:
        """
        Internal method to process a flexible node configuration into a strict node configuration.
        
        Parameters
        ----------
        config_item : :obj:`~HeteroSymNN.types.FlexibleNodeConfig`
            Flexible node configuration (string or tuple).
            
        Returns
        -------
        :obj:`~HeteroSymNN.types.NodeConfig`
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
        
        Child class of :obj:`~HeteroSymNN.Core.Nets.Dense`.
        
        Parameters
        ----------
        num_layers : int
            Number of layers in the network.
        nodes_per_layer : list[int]
            List containing the number of nodes in each layer.
        layer_configs : list[:obj:`~HeteroSymNN.types.FlexibleNodeConfig`]
            List containing the flexible activation configuration for each layer.

        Returns
        -------
        list[list[:obj:`~HeteroSymNN.types.NodeConfig`]]
            Detailed activation configuration for each node in each layer.
        """
        final_config = []
        for i in range(num_layers):
            layer_conf_raw = layer_configs[i]
            num_nodes = nodes_per_layer[i]
            
            node_conf = self._process_node_config(layer_conf_raw)
            
            final_config.append([node_conf] * num_nodes)
            
        return final_config
    

class MLP(Dense):
    """
    High-level class for creating dense neural networks with uniform activation functions across hidden layers.
    
    Parameters
    ----------
    nodes_structure : list[int]
        List containing the number of nodes in each layer including input and output layers.
    activation : :obj:`~HeteroSymNN.types.FlexibleNodeConfig`, optional
        Activation function configuration for hidden layers. Can be a string (activation name) or a tuple (activation name, parameters dictionary)., by default "relu
    output_activation : :obj:`~HeteroSymNN.types.FlexibleNodeConfig`, optional
        Activation function configuration for the output layer. Can be a string (activation name) or a tuple (activation name, parameters dictionary)., by default "num
    initializer : Optional[:obj:`~HeteroSymNN.Core.Nets.initializers.Initializer`], optional
        Initializer to use for initializing weights and biases. If not provided, :obj:`~HeteroSymNN.Core.Nets.initializers.HeNormal` will be used., by default None
    learning_rate : float, optional
        Learning rate for the network., by default 0.001
    training_mode : Literal["batch", "mini-batch", "stochastic"], optional
        Training mode to use during training. In case of "batch" or "stochastic" the batch size attribute will be ignored., by default "stochastic"
    batch_size : int, optional
        Batch size to use during training. In the case of using "stochastic" or "batch" training mode this attribute will be ignored and in training time the batch size will be set to 1 or to the full dataset size respectively., by default 32
    loss_function : :obj:`~HeteroSymNN.Core.Nets.losses.Loss`, optional
        Loss function to use for training. If not provided, :obj:`~HeteroSymNN.Core.Nets.losses.MSELoss` will be used., by default None
    optimizer : :obj:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`, optional
        Optimizer to use for training. If not provided, :obj:`~HeteroSymNN.Core.Nets.optimizers.AdamOptimizer` will be used., by default None
    num_treaning_iter : int, optional
        Number of training iterations (epochs)., by default 1000
    
    Attributes
    ----------
    num_treaning_iterations : int, read-write
        Number of training iterations (epochs) for the network.
    learning_mode : str, read-write
        Learning mode of the network. Currently only "Static" is supported.
    training_mode : Literal["batch", "mini-batch", "stochastic"], read-write
        Training mode to use during training. When seting it to "mini-batch" from "stochastic" or "batch" the batch size that will be used is the one stored in the attribute batch_size.
    batch_size : int, read-write
        Batch size to use during training.
    histogram_losses : list[float], read-only
        List of loss values recorded at each epoch during training.
    num_complited_train_iterations : int, read-only
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
        ...     batch_size=64,
        ...     training_mode="mini-batch"
        ... )
    """
    def __init__(self, nodes_structure: list[int], activation: FlexibleNodeConfig = "relu", output_activation: FlexibleNodeConfig = "num",initializer: Optional[InitC.Initializer] = None,
                 learning_rate: float = 0.001, training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic",
                 batch_size: int = 32, loss_function: Optional[lossC.Loss] = None, optimizer: Optional[OptiC.Optimizer] = None, num_treaning_iter: int = 1000):

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
            training_mode=training_mode,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer=optimizer,
            num_treaning_iter=num_treaning_iter
        )