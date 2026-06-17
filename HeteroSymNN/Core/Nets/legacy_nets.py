from __future__ import annotations
from typing import Optional,Literal

from ...types import NodeConfig,LayerValues,FlexibleNodeConfig
from .linear_net import HeteroLinearNet, LinearNet, MLP
from .. import losses as lossC, optimizers as OptiC, initializers as InitC
from ...exceptions import  HeteroSymNNDeprecationWarning
import warnings

class ConfigurableNN(HeteroLinearNet):
    """
        Legacy class for a linear neural network with customizable activation functions per neuron.
        
        Parameters
        ----------
        nodes_structure : list[int]
            List with the number of nodes per layer including input and output layers.
        detailed_activations : list[list[:type:`~HeteroSymNN.types.NodeConfig`]]
            List of lists containing the activation configuration for each node in each layer.
        initial_values : Optional[list[:type:`~HeteroSymNN.types.LayerValues`]], optional
            Optional list of initial values for each layer. If not provided, weights and biases will be initialized using the specified initializer., by default None
        initializer : Optional[:class:`~HeteroSymNN.Core.Nets.initializers.Initializer`], optional
            Initializer to use for weights and biases if initial_values is not provided. Most pass an instance of :class:`~HeteroSymNN.Core.Nets.initializers.Initializer` and the default uses :class:`~HeteroSymNN.Core.Nets.initializers.HeNormal`, value by default is None.
        learning_rate : float, optional
            Learning rate for the network. In the case that a custom optimizer is provided with its own learning rate this value will be overwritten., by default 0.001
        batch_size : int, optional
            Batch size to use during training, by default 32 if training_mode is "mini-batch", 1 if "stochastic" and size of the dataset if "batch".
        training_mode : Literal["batch", "mini-batch", "stochastic"], optional
            Training mode to use during training. Options are "batch", "mini-batch", and "stochastic". By default "mini-batch".
        learning_mode : str, optional
            Learning mode of the network. Currently only "Static" is supported., by default "Static"
        loss_function : :class:`~HeteroSymNN.Core.Nets.losses.Loss`, optional
            Loss function to use during training. Must be an instance of :class:`~HeteroSymNN.Core.Nets.losses.Loss`. If not provided, ::class:`~HeteroSymNN.Core.Nets.losses.MSELoss` will be used., value by default is None.
        optimizer : Optional[:class:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`], optional
            Optimizer to use for updating the network parameters. Must be an instance of :class:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`. If not provided, :class:`~HeteroSymNN.Core.Nets.optimizers.AdamOptimizer` will be used.,value by default is None.
        num_treaning_iter : int, optional
            Number of Epochs to use during training, by default 1000

        Attributes
        ----------
        num_training_epochs : int, read-write
            Number of training iterations (epochs) for the network.
        training_mode : Literal["batch", "mini-batch", "stochastic"], read-write
            Training mode to use during training. When seting it to "mini-batch" from "stochastic" or "batch" the batch size that will be used is the one stored in the attribute batch_size.
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
        >>> from HeteroSymNN.Core.Nets.neural_nets import ConfigurableNN
        >>> CNN = ConfigurableNN(
        ...     nodes_structure=[3, 5, 2],
        ...     detailed_activations=[
        ...         [("relu", {}), ("relu", {}), ("relu", {}), ("relu", {}), ("relu", {})],
        ...         [("sigmoid", {}), ("sigmoid", {})]
        ...     ],
        ...     learning_rate=0.01,
        ...     batch_size=16,
        ...     training_mode="mini-batch"
        ... )

        
        >>> from HeteroSymNN.Core.Nets import losses as lossC, optimizers as OptiC, initializers as InitC
        >>> custom_loss = lossC.CrossEntropyLoss()
        >>> custom_optimizer = OptiC.SGDOptimizer(learning_rate=0.01)
        >>> custom_initializer = InitC.XavierUniform()
        >>> CNN = ConfigurableNN(
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
    def __init__(self, nodes_structure: list[int], detailed_activations: list[list[NodeConfig]],initial_values: Optional[list[LayerValues]] = None,initializer: Optional[InitC.Initializer] = None,
                 learning_rate: float = 0.001, batch_size: int = 32, training_mode: Literal["batch", "mini-batch", "stochastic"] = "mini-batch", learning_mode: str = "Static",
                 loss_function: Optional[lossC.Loss] = None, optimizer: Optional[OptiC.Optimizer] = None, num_treaning_iter: int = 1000):
        warnings.warn(
            HeteroSymNNDeprecationWarning(
                deprecated_feature="ConfigurableNN",
                replacement="HeteroLinearNet",
                version_removal="0.4.0"
            ),
            stacklevel=2
        )
        if initializer is not None and not isinstance(initializer, list):
            initializer = [initializer] * (len(nodes_structure) - 1)
            
        if training_mode == "stochastic":
            batch_size = 1
        elif training_mode == "batch":
            batch_size = -1
            
        super().__init__(
            num_inputs=nodes_structure[0], 
            detailed_activations=detailed_activations, 
            initializer=initializer, 
            learning_rate=learning_rate, 
            batch_size=batch_size, 
            loss_function=loss_function, 
            optimizer=optimizer, 
            num_training_iter=num_treaning_iter
        )
        self.training_mode = training_mode
        if initial_values is not None:
            dict_init_vals:dict[int,dict[str,list]] = {}
            for i in range(len(initial_values)):
                dict_init_vals[i] = {"weights":initial_values[i][1], "biases":initial_values[i][0],"connection_mask":initial_values[i][2]}
            self.set_parameters(dict_init_vals)


class FlexibleNN(LinearNet):
    """
    Legacy class for creating linear_net neural networks with diferent activation functions per layer.
    
    Parameters
    ----------
    nodes_structure : list[int]
        List containing the number of nodes in each layer including input and output layers.
    activation_config : list[:type:`~HeteroSymNN.types.FlexibleNodeConfig`]
        List containing the activation configuration for each layer. Each element can be a string (activation name) or a tuple (activation name, parameters dictionary).
    initial_values : Optional[list[:type:`~HeteroSymNN.types.LayerValues`]], optional
        List of initial values for weights and biases for each layer. If not provided, weights and biases will be initialized using the specified initializer., by default None
    initializer : Optional[:class:`~HeteroSymNN.Core.Nets.initializers.Initializer`], optional
        Initializer to use for initializing weights and biases. If not provided, :class:`~HeteroSymNN.Core.Nets.initializers.HeNormal` will be used., by default None
    learning_rate : float, optional
        Learning rate for the network., by default 0.001
    learning_mode : str, optional
        Learning mode of the network. Currently only "Static" is supported., by default "Static"
    training_mode : Literal["batch", "mini-batch", "stochastic"], optional
        Training mode to use during training. In case of "batch" or "stochastic" the batch size attribute will be ignored., by default "stochastic"
    batch_size : int, optional
        Batch size to use during training. In the case of using "stochastic" or "batch" training mode this attribute will be ignored and in training time the batch size will be set to 1 or to the full dataset size respectively., by default 32
    loss_function : :class:`~HeteroSymNN.Core.Nets.losses.Loss`, optional
        Loss function to use for training. If not provided, :class:`~HeteroSymNN.Core.Nets.losses.MSELoss` will be used., by default None
    optimizer : :obj:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`, optional
        Optimizer to use for training. If not provided, :obj:`~HeteroSymNN.Core.Nets.optimizers.AdamOptimizer` will be used., by default None
    num_treaning_iter : int, optional
        Number of training iterations (epochs)., by default 1000

    Attributes
    ----------
    num_training_epochs : int, read-write
        Number of training iterations (epochs) for the network.
    training_mode : Literal["batch", "mini-batch", "stochastic"], read-write
        Training mode to use during training. When seting it to "mini-batch" from "stochastic" or "batch" the batch size that will be used is the one stored in the attribute batch_size.
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
        >>> from HeteroSymNN.Core.Nets.neural_nets import FlexibleNN
        >>> FNN = FlexibleNN(
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
                 learning_rate: float = 0.001, learning_mode: str = "Static", training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic", batch_size: int = 32,
                 loss_function: Optional[lossC.Loss] = None, optimizer: Optional[OptiC.Optimizer] = None, num_treaning_iter: int = 1000):
        warnings.warn(
            HeteroSymNNDeprecationWarning(
                deprecated_feature="FlexibleNN",
                replacement="LinearNet",
                version_removal="0.4.0"
            ),
            stacklevel=2
        )
        if training_mode == "stochastic":
            batch_size = 1
        elif training_mode == "batch":
            batch_size = -1
            
        super().__init__(
            nodes_structure=nodes_structure, 
            activation_config=activation_config,
            initializer=initializer,
            learning_rate=learning_rate, 
            batch_size=batch_size,
            loss_function=loss_function, 
            optimizer=optimizer, 
            num_training_iter=num_treaning_iter
        )
        self.training_mode = training_mode
        if initial_values is not None:
            dict_init_vals:dict[int,dict[str,list]] = {}
            for i in range(len(initial_values)):
                dict_init_vals[i] = {"weights":initial_values[i][1], "biases":initial_values[i][0],"connection_mask":initial_values[i][2]}
            self.set_parameters(dict_init_vals)

class SimpleNN(MLP):
    """
    Legacy class for creating linear_net neural networks with uniform activation functions across hidden layers.
    
    Parameters
    ----------
    nodes_structure : list[int]
        List containing the number of nodes in each layer including input and output layers.
    activation : :type:`~HeteroSymNN.types.FlexibleNodeConfig`, optional
        Activation function configuration for hidden layers. Can be a string (activation name) or a tuple (activation name, parameters dictionary)., by default "relu
    output_activation : :type:`~HeteroSymNN.types.FlexibleNodeConfig`, optional
        Activation function configuration for the output layer. Can be a string (activation name) or a tuple (activation name, parameters dictionary)., by default "num
    initializer : Optional[:class:`~HeteroSymNN.Core.Nets.initializers.Initializer`], optional
        Initializer to use for initializing weights and biases. If not provided, :class:`~HeteroSymNN.Core.Nets.initializers.HeNormal` will be used., by default None
    learning_rate : float, optional
        Learning rate for the network., by default 0.001
    learning_mode : str, optional
        Learning mode of the network. Currently only "Static" is supported., by default "Static"
    training_mode : Literal["batch", "mini-batch", "stochastic"], optional
        Training mode to use during training. In case of "batch" or "stochastic" the batch size attribute will be ignored., by default "stochastic"
    batch_size : int, optional
        Batch size to use during training. In the case of using "stochastic" or "batch" training mode this attribute will be ignored and in training time the batch size will be set to 1 or to the full dataset size respectively., by default 32
    loss_function : :class:`~HeteroSymNN.Core.Nets.losses.Loss`, optional
        Loss function to use for training. If not provided, :class:`~HeteroSymNN.Core.Nets.losses.MSELoss` will be used., by default None
    optimizer : :class:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`, optional
        Optimizer to use for training. If not provided, :class:`~HeteroSymNN.Core.Nets.optimizers.AdamOptimizer` will be used., by default None
    num_treaning_iter : int, optional
        Number of training iterations (epochs)., by default 1000

    Attributes
    ----------
    num_training_epochs : int, read-write
        Number of training iterations (epochs) for the network.
    training_mode : Literal["batch", "mini-batch", "stochastic"], read-write
        Training mode to use during training. When seting it to "mini-batch" from "stochastic" or "batch" the batch size that will be used is the one stored in the attribute batch_size.
    batch_size : int, read-write
        Batch size to use during training.s
    history_losses : list[float], read-only
        List of loss values recorded at each epoch during training.
    num_completed_train_iterations : int, read-only
        Number of completed training steps.
    num_completed_epochs : int, read-only
        Number of completed training epochs.

    Examples
    --------
        >>> from HeteroSymNN.Core.Nets.neural_nets import SimpleNN
        >>> SNN = SimpleNN(
        ...     nodes_structure=[4, 8, 3],
        ...     activation="tanh",
        ...     output_activation="softmax",
        ...     learning_rate=0.01,
        ...     batch_size=64,
        ...     training_mode="mini-batch"
        ... )
    """
    def __init__(self, nodes_structure: list[int], activation: FlexibleNodeConfig = "relu", output_activation: FlexibleNodeConfig = "num",initializer: Optional[InitC.Initializer] = None,
                 learning_rate: float = 0.001, learning_mode: str = "Static",training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic",
                 batch_size: int = 32, loss_function: Optional[lossC.Loss] = None, optimizer: Optional[OptiC.Optimizer] = None, num_treaning_iter: int = 1000):
        warnings.warn(
            HeteroSymNNDeprecationWarning(
                deprecated_feature="SimpleNN",
                replacement="MLP",
                version_removal="0.4.0"
            ),
            stacklevel=2
        )
        if training_mode == "stochastic":
            batch_size = 1
        elif training_mode == "batch":
            batch_size = -1
            
        super().__init__(
            nodes_structure=nodes_structure, 
            activation=activation, 
            output_activation=output_activation,
            initializer=initializer,
            learning_rate=learning_rate, 
            batch_size=batch_size, 
            loss_function=loss_function, 
            optimizer=optimizer, 
            num_training_iter=num_treaning_iter
        )
        self.training_mode = training_mode