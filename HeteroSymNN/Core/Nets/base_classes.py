from __future__ import annotations
import numpy as np
from typing import Literal, Union, Optional, Any, Sequence
import warnings
import inspect
import os
import datetime
import json
import zipfile
from pathlib import Path
import io

from ... import __version__
from ...Backend import hardware as HW
from ...Backend.validators import _validate_gpu_id
from .. import losses as lossC, optimizers as OptiC, initializers as InitC
from ..layers import BaseLayer
from ...types import LayerConstruction,NodeConfig,BackendArray,ConstantToUpdate
from ...exceptions import NetworkStructureError, LayerConfigurationError, MethodMigrationError, ShapeMismatchError,LoadingError,BackendNotAvailableWarning,PerformanceWarning, SavingError, PathError,HeteroSymNNValueError
from ...error_handlers import clean_traceback
from ...config import settings

__all__ = ["BaseNetwork"]

class BaseNetwork:
    """
        Base class of all networks with customizable architecture, activation functions, and training parameters.
        
        Parameters
        ----------
        num_inputs : int
            Number of input units to the network.
        network_structure : list[type[:class:`~HeteroSymNN.Core.layers.BaseLayer`]]
            List with the type of layers that will be used for each step of the network.
        extra_layer_parameters : list[dict[str,Any]]
            list of dictionaries with the extra parameters that the layer need to work correctly.
        detailed_activations : list[list[:type:`~HeteroSymNN.types.NodeConfig`]]
            List of lists containing the activation configuration for each node in each layer.
        initializers : list[:class:`~HeteroSymNN.Core.Nets.initializers.Initializer`], optional
            List of initializers to use for weights and biases if initial_values is not provided. Must pass instances of :class:`~HeteroSymNN.Core.Nets.initializers.Initializer` and if value is left as None it will use :class:`~HeteroSymNN.Core.Nets.initializers.HeNormal`.
        learning_rate : float, optional
            Learning rate for the network. In the case that a custom optimizer is provided with its own learning rate this value will be overwritten., by default 0.001
        batch_size : int, optional
            Batch size to use during training, by default 32. If set to -1, it uses the full dataset size for batch training.
        loss_function : :class:`~HeteroSymNN.Core.Nets.losses.Loss`, optional
            Loss function to use during training. Must be an instance of :class:`~HeteroSymNN.Core.Nets.losses.Loss`. If value is left as None, :class:`~HeteroSymNN.Core.Nets.losses.MSELoss` will be used.
        optimizer : Optional[:class:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`], optional
            Optimizer to use for updating the network parameters. Must be an instance of :class:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`. If value is left as None, :class:`~HeteroSymNN.Core.Nets.optimizers.AdamOptimizer` will be used.
        num_epochs: int, optional
            Number of Epochs to use during training, by default 1000
        gpu_id: int, optional
            ID of the GPU to allocate the network on initially, by default 0.

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
        >>> from HeteroSymNN.Core.Nets import BaseNetwork
        >>> from HeteroSymNN.Core.layers import LinearLayer
        >>> NN = BaseNetwork(
        ...     num_inputs=3,
        ...     network_structure=[LinearLayer, LinearLayer],
        ...     extra_layer_parameters=[{}, {}],
        ...     detailed_activations=[
        ...         [("relu", {}), ("relu", {}), ("relu", {}), ("relu", {}), ("relu", {})],
        ...         [("sigmoid", {}), ("sigmoid", {})]
        ...     ],
        ...     learning_rate=0.01,
        ...     batch_size=16,
        ... )

        
        >>> from HeteroSymNN.Core.Nets import BaseNetwork
        >>> from HeteroSymNN.Core.layers import LinearLayer
        >>> from HeteroSymNN.Core import initializers,optimizers,losses
        >>> custom_loss_func = losses.BinaryCrossEntropy()
        >>> custom_optimizer = optimizers.SgdOptimizer(0.01)
        >>> custom_initializers = [initializers.HeNormal(0.8),initializers.XavierUniform(1)]
        >>> NN = BaseNetwork(
        ...     num_inputs=3,
        ...     network_structure=[LinearLayer, LinearLayer],
        ...     extra_layer_parameters=[{}]*2,
        ...     detailed_activations=[
        ...         [("relu", {}), ("relu", {}), ("relu", {}), ("relu", {}), ("relu", {})],
        ...         [("sigmoid", {}), ("sigmoid", {})]
        ...     ],
        ...     initializers=custom_initializers,
        ...     optimizer=custom_optimizer,
        ...     loss_function=custom_loss_func
        ... )       
    """
    @clean_traceback
    def __init__(self, num_inputs:int, network_structure: list[type[BaseLayer]],extra_layer_parameters:list[dict[str,Any]], detailed_activations: list[list[NodeConfig]],initializers: Optional[list[InitC.Initializer]] = None,
                 learning_rate: float = 0.001,batch_size: int = 32, loss_function: Optional[lossC.Loss] = None, optimizer: Optional[OptiC.Optimizer] = None, num_epochs: int = 1000,
                 gpu_id: int = 0):
        
        self._CALCULATION_MANAGER = settings.default_manager
        self._ASNUMPY = settings.default_asnumpy
        self.num_completed_train_iterations = 0
        self.num_completed_epochs = 0
        self._input_gradient = None

        
        if len(network_structure) < 1:
            raise NetworkStructureError("network_structure most have at least 1 value, the class of the  output layer.")
        
        num_layers = len(network_structure)
        if (len(detailed_activations) != num_layers):
            raise NetworkStructureError(f"The structure is defined as {num_layers} layers, but 'detailed_activations' has {len(detailed_activations)} elements.")

        if (len(extra_layer_parameters) != num_layers):
            raise NetworkStructureError(f"The structure is defined as {num_layers} layers, but 'extra_layer_parameters' has {len(extra_layer_parameters)} elements.")
        
        self._BATCH_SIZE = batch_size
        _validate_gpu_id(gpu_id)
        self._GPU_ID = gpu_id
        self._DEFAULT_FLOAT_TYPE = settings.default_dtype
        self._CURRENT_DEVICE = "CPU"
        self._CURRENT_LOCATION = "host"
        self._COMPUTATIONAL_METHOD = settings.default_compute_method
        self.num_training_epochs = num_epochs
        self._NETWORK_STRUCTURE = network_structure
        self._init_density_of_mask = 1.0
        self._initializers = []


        if (initializers is None):
            initializers = []
            for _ in range(num_layers):
                initializers.append(InitC.HeNormal())
        
        if (len(initializers) != num_layers):
            raise NetworkStructureError(f"There were given {len(initializers)} initializers, but the structure is defined as {num_layers} layers.")
        else:
            self._initializers = initializers

        if (loss_function == None):
            self._LOSS_FUNCTION:lossC.Loss = lossC.MSELoss(self._COMPUTATIONAL_METHOD,self._GPU_ID)
        else:
            self._LOSS_FUNCTION = loss_function
            temp_result = self._LOSS_FUNCTION._change_COMPUTATIONAL_METHOD(self._COMPUTATIONAL_METHOD,self._GPU_ID)
            if (temp_result != self._COMPUTATIONAL_METHOD):
                raise MethodMigrationError(f"The loss function computational method ({temp_result}) couldn't be sync with the main network computational method ({self._COMPUTATIONAL_METHOD}).")
                

        if not(optimizer is None):
            self._UPDATE_METHOD = optimizer
            if (self._UPDATE_METHOD.learning_rate is None):
                self._UPDATE_METHOD.learning_rate = learning_rate
        else:
            self._UPDATE_METHOD:OptiC.Optimizer = OptiC.AdamOptimizer(learning_rate,self._COMPUTATIONAL_METHOD.split("_")[0],device_id=self._GPU_ID)
        
        self._LAYERS: list[BaseLayer] = []

        num_nodes = num_inputs
        for i in range(num_layers):
            num_inputs = num_nodes
            num_nodes = len(detailed_activations[i])
            layer_class:type[BaseLayer] = network_structure[i]

            if not(callable(layer_class)):
                raise NetworkStructureError(f"Object given for the creation of layer {i} is not a callable class.")

            if not(issubclass(layer_class,BaseLayer)):
                raise NetworkStructureError(f"Class given for the creation of layer {i} is not a subclass of BaseLayer.")
            
            node_configs = detailed_activations[i]
            extra_param = extra_layer_parameters[i]
            
            sig = inspect.signature(layer_class)

            required_arguments = set()
            all_accepted_arguments = set()

            for name, param in sig.parameters.items():
                all_accepted_arguments.add(name)
                if param.default == inspect.Parameter.empty and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    required_arguments.add(name)

            required_arguments.discard('self')
            all_accepted_arguments.discard('self')

            total_provided = set(extra_param.keys()) | {"num_inputs", "layer_configuration", "batch_size", "Gpu_id"}

            missing = required_arguments - total_provided
            wrong = total_provided - all_accepted_arguments

            if missing:
                raise LayerConfigurationError(
                    f"Layer {i} is missing {len(missing)} required parameters: {missing}."
                )

            if wrong:
                raise LayerConfigurationError(
                    f"Layer {i} received {len(wrong)} unrecognized parameters: {wrong}."
                )

                                              
            layer_init = self._initializers[i]

            layer_config: LayerConstruction = (node_configs, layer_init)
            try:
                self._LAYERS.append(layer_class(num_inputs, layer_config, self._BATCH_SIZE, self._GPU_ID,**extra_param))
            except LayerConfigurationError as e:
                raise LayerConfigurationError(f"Error in Layer {i}: {e}")

        self.history_losses = []    
    
    @classmethod
    @clean_traceback
    def from_config(cls, general_configs: dict[str,Any], registry_module) -> "BaseNetwork":
        """
        Machine Resurrection Method.
        Bypasses __init__ to allocate memory for the specific network subclass and populates the physics directly.

        Parameters
        ----------
        general_configs: dict[str,any]
            json in dictionary form of the config.json file inside .symnn files.
        registry_module: :class:`~HeteroSymNN.Core.Nets.registries._Registry`
            registry intance with all the class dictionaries for the instansing of the used classes.
            
        Returns
        -------
        :class:`~HeteroSymNN.Core.Nets.BaseNetwork`
            instanced object of the network class.
        """
        
        instance = cls.__new__(cls)
        
        instance._CALCULATION_MANAGER = settings.default_manager
        instance._ASNUMPY = settings.default_asnumpy
        instance._DEFAULT_FLOAT_TYPE = settings.default_dtype
        instance._CURRENT_DEVICE = "CPU"
        instance._CURRENT_LOCATION = "host"
        instance._COMPUTATIONAL_METHOD = settings.default_compute_method
        instance._GPU_ID = 0
        

        architecture_config = general_configs.get('architecture', {})
        metadata = general_configs.get('metadata', {})

        instance._NETWORK_STRUCTURE = []
        instance._BATCH_SIZE = architecture_config.get('batch_size', 32)
        instance.num_training_epochs = architecture_config.get('num_training_epochs', 1000)
        
        instance.num_completed_train_iterations = metadata.get('num_completed_train_iterations', 0)
        instance.num_completed_epochs = metadata.get('total_epochs_iterations', 0)
        instance.history_losses = []
        

        loss_fn_config = architecture_config['loss_config']
        loss_class_name:str = loss_fn_config['class_name']
        if loss_class_name not in registry_module.loss_fn_map:
            raise LoadingError(f"Unknown loss function: {loss_class_name}.")
        loss_fn = registry_module.loss_fn_map[loss_class_name](**{k: v for k, v in loss_fn_config.items() if k != 'class_name'}) 
        

        optimizer_config = general_configs['optimizer_config']
        opt_class_name:str = optimizer_config['class_name']
        if (opt_class_name not in registry_module.optimizers_map):
            raise LoadingError(f"Unknown optimizer: {opt_class_name}.")
        optimizer = registry_module.optimizers_map[opt_class_name](**{k: v for k, v in optimizer_config.items() if k != 'class_name'})
        
        instance._LOSS_FUNCTION = loss_fn
        instance._UPDATE_METHOD = optimizer
        

        instance._LAYERS = []
        layers_configs_dict = architecture_config.get('layer_configs', {}) # Matched the key!
        
        sorted_layer_keys = sorted(layers_configs_dict.keys(), key=lambda k: int(k.split('_')[1]))
        
        for layer_key in sorted_layer_keys:
            layer_cfg = layers_configs_dict[layer_key]
            
            layer_class_name = layer_cfg['layer_type']
            LayerClass = registry_module.layers_map[layer_class_name]
            
            init_cfg = layer_cfg.get('initializer')
            rebuilt_initializer = None
            if init_cfg:
                InitClass = registry_module.initializers_map[init_cfg.pop('class_name')]
                rebuilt_initializer = InitClass(**init_cfg)
            
            layer_construction = (layer_cfg['layer_node_configs'], rebuilt_initializer)
            
            rebuilt_layer = LayerClass(
                num_inputs=layer_cfg['num_inputs'],
                layer_configuration=layer_construction,
                batch_size=instance._BATCH_SIZE,
                Gpu_id=instance._GPU_ID
            )
            instance._LAYERS.append(rebuilt_layer)
            instance._NETWORK_STRUCTURE.append(LayerClass)
            
        return instance
    
    @property
    def input_gradient(self)->np.ndarray:
        """
        Property to get the input gradient of the network. Read-only.
        
        Returns
        -------
        np.ndarray
        """
        return self._ASNUMPY(self._input_gradient)
    
    @property
    def layers(self)->Sequence[BaseLayer]:
        """
        Property to get the layers of the network. Read-only.
        
        Returns
        -------
        list[:class:`~HeteroSymNN.Core.Nets.layers.BaseLayer`]
        """
        return self._LAYERS
    
    @property
    def network_structure(self)->Sequence[int]:
        """
        Property to get the number of nodes and the callers for each layer in the network. Read-only.
        
        Returns
        -------
        list[int]
        """
        return self._NETWORK_STRUCTURE
    
    @property
    def gpu_id(self)->int:
        """
        Property to get the current GPU ID being used by the network. Read-only.
        
        For setting a new GPU ID, use the :meth:`set_gpu_id` method.

        Returns
        -------
        int
        """
        return self._GPU_ID
    
    @property
    def batch_size(self)->int:
        """
        Property to get the current batch size being used by the network in case of mini-batch training. Read-only.
        
        Returns
        -------
        int
        """
        return self._BATCH_SIZE
    
    @property
    def current_device(self)->Literal["CPU","GPU"]:
        """
        Property to get the current device where the network parameters are located. Read-only.

        For changing location of the network parameters use the :meth:`to` method.
        
        Returns
        -------
        Literal["CPU","GPU"]
        """
        return self._CURRENT_DEVICE
    
    @property
    def current_location(self)->Literal["host","device"]:
        """
        Property to get the current logical location where the network parameters are located. Read-only.

        For changing location of the network parameters use the :meth:`to` method.
        
        Returns
        -------
        Literal["host","device"]
        """
        return self._CURRENT_LOCATION
    
    @property
    def computational_method(self)->Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]:
        """
        Property to get the current computational method being used by the network. Read-only.

        To change the computational method use the :meth:`set_backend` method.
        
        Returns
        -------
        Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]
        """
        return self._COMPUTATIONAL_METHOD

    @property
    def optimizer(self)->OptiC.Optimizer:
        """
        Property to get the current optimizer being used by the network. Read-only.

        Returns
        -------
        :class:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`
        """
        return self._UPDATE_METHOD

    @property
    def loss_function(self)->lossC.Loss:
        """
        Property to get the current loss function being used by the network. Read-only.

        Returns
        -------
        :class:`~HeteroSymNN.Core.Nets.losses.Loss`
        """
        return self._LOSS_FUNCTION
    
    @property
    def initializer(self)->list[InitC.Initializer]:
        """
        Property to get the current initializer being used by the network. Read-only.

        Returns
        -------
        list[:class:`~HeteroSymNN.Core.Nets.initializers.Initializer`]
        """
        return self._initializers

    @property
    def learning_rate(self)->float:
        """
        Property to get the current learning rate of the network. Read-write.
        
        Returns
        -------
        float
        """
        return self._UPDATE_METHOD.learning_rate
    
    @learning_rate.setter
    def learning_rate(self,new_learning_rate:float):
        self._UPDATE_METHOD.learning_rate = new_learning_rate

    @clean_traceback
    def set_gpu_id(self,new_id:int)->None:
        """
        Method to set a new GPU ID for the network. This will change the device of the network parameters if currently on GPU.

        Parameters
        ----------
        new_id : int
            New GPU ID to set.
        
        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.InvalidDeviceIDError`
            If the Id for the new GPU is greater than the number of available GPUs.
        """
        if (self._GPU_ID != new_id):
            _validate_gpu_id(new_id)
            self.to("host")
            self._GPU_ID = new_id
            self._LOSS_FUNCTION.set_gpu_id(self._GPU_ID)
            self._UPDATE_METHOD.set_gpu_id(self._GPU_ID)
            for layer in self._LAYERS:
                layer.set_gpu_id(new_id)

    @clean_traceback
    def _change_COMPUTATIONAL_METHOD(self,new_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],gpu_id:int = None)->None:
        """
        Change the computational method used by the network. This will also change the device of the network parameters if needed.

        Parameters
        ----------
        new_method : Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]
            New computational method to set.
        gpu_id : int, optional
            GPU ID to use if the new method is "GPU_CUDA". If not provided, the current GPU ID of the network will be used., by default None
        
       Raises
        ------
        :exc:`~HeteroSymNN.exceptions.HeteroSymNNValueError`
            If the new method is not one of "GPU_CUDA", "CPU_JIT", or "CPU_PYTHON".
        :exc:`~HeteroSymNN.exceptions.BackendNotAvailableError`
            If trying to set "GPU_CUDA" without a valid GPU or "CPU_JIT" without a valid C++ compiler when strict warnings mode is enabled. If not enabled, it will fallback to the next available method and throw a warning.
        :exc:`~HeteroSymNN.exceptions.MethodMigrationError`
            In the case the loss function or any of the layers could not change to the new computational method will show a runtime error.
        
        .. Warning::
            When changing computational methods it will force a kernel recompilation for all layers and reallocation of all parameters. Depending on the size of the network this could take a while.
        """
        new_method = new_method.upper()
        try_method = new_method
        msg_extra = ""
        if not(new_method in ["GPU_CUDA","CPU_JIT","CPU_PYTHON"]):
            raise HeteroSymNNValueError("tried to change the computational method to something that isn't GPU_CUDA, CPU_JIT or CPU_PYTHON")
        
        if (gpu_id == None):
            gpu_id = self._GPU_ID
        else:
            _validate_gpu_id(gpu_id)

        if ((new_method == "GPU_CUDA") and not(new_method in settings.available_methods)):
            msg_extra = ", but no GPU is available."
            new_method = "CPU_PYTHON"

        if ((new_method == "CPU_JIT")):
                msg_extra = ", but currently is not available"
                new_method = "CPU_PYTHON"

        if (try_method != new_method):
                warnings.warn(f"Tried to change to use '{try_method}'{msg_extra}. {new_method} is required",BackendNotAvailableWarning,stacklevel=2)

        if(new_method != self._COMPUTATIONAL_METHOD):
                warnings.warn(
                f"Initiating engine migration to {new_method}. This requires JIT recompilation and memory transfers.",
                PerformanceWarning,
                stacklevel=2
                )
                self.to("host")
                self._COMPUTATIONAL_METHOD = new_method
                self._GPU_ID = gpu_id
                if ("CPU" in new_method):
                    self._CALCULATION_MANAGER = np
                    self._ASNUMPY = np.array
                elif ("GPU" in new_method):
                    self._CALCULATION_MANAGER = HW.cp
                    self._ASNUMPY = HW.cp.asnumpy
                
                temp_result = self._LOSS_FUNCTION._change_COMPUTATIONAL_METHOD(self._COMPUTATIONAL_METHOD,self._GPU_ID)
                if (temp_result != self._COMPUTATIONAL_METHOD):
                    raise MethodMigrationError (f"Loss function couldn't be synconized with the new computational method {self._COMPUTATIONAL_METHOD}. Loss function is stuck in {temp_result}")
                self._UPDATE_METHOD._change_COMPUTATIONAL_DEVICE(self._COMPUTATIONAL_METHOD.split("_")[0])
                laye_calc_method = []
                for layer in self._LAYERS:
                    laye_calc_method.append(layer._change_COMPUTATIONAL_METHOD(new_method,gpu_id))
                
                expected = [new_method]*len(self._LAYERS)
                if (laye_calc_method != expected):
                    raise MethodMigrationError(f"Couldn't change the computational method due to one or more layers couldn't change. Layer methods list: {laye_calc_method}")

    @clean_traceback
    def set_backend(self,backend:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],gpu_id:int = None)->None:
        """
        Change the computational method used by the network. This will also change the device of the network parameters if needed.

        Parameters
        ----------
        backend : Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]
            New computational method to set.
        gpu_id : int, optional
            GPU ID to use if the new method is "GPU_CUDA". If not provided, the current GPU ID of the network will be used., by default None
        
        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.HeteroSymNNValueError`
            If the new method is not one of "GPU_CUDA", "CPU_JIT", or "CPU_PYTHON".
        :exc:`~HeteroSymNN.exceptions.BackendNotAvailableError`
            If trying to set "GPU_CUDA" without a valid GPU or "CPU_JIT" without a valid C++ compiler when strict warnings mode is enabled. If not enabled, it will fallback to the next available method and throw a warning.
        :exc:`~HeteroSymNN.exceptions.MethodMigrationError`
            In the case the loss function or any of the layers could not change to the new computational method will show a runtime error.
        
        .. Warning::
            When changing computational methods it will force a kernel recompilation for all layers and reallocation of all parameters. Depending on the size of the network this could take a while.
        """
        self._change_COMPUTATIONAL_METHOD(new_method=backend, gpu_id=gpu_id)

    @clean_traceback
    def to(self, location:Literal["host","device"])->None:
        """
        Change logical location of the network parameters to the specified location.

        Parameters
        ----------
        location : Literal["host","device"]
            Location to move the network parameters to.
        
        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.HeteroSymNNValueError`
            If the location is not "host" or "device".
        """
        location = location.lower()
        if not(location in ["host","device"]):
            raise HeteroSymNNValueError("Specify location is not host or device.")

        if(location != self._CURRENT_LOCATION):
                self._CURRENT_LOCATION = location
                
                target_hardware = "CPU"
                if location == "device":
                     if "GPU" in self._COMPUTATIONAL_METHOD:
                         target_hardware = "GPU"
                self._CURRENT_DEVICE = target_hardware

                self._to(location)

    def cast_arrays(self, *tensors: np.ndarray, gpu_id: int = None)->Union[BackendArray,tuple[BackendArray]]:
        """
        Method to cast one or more arrays to the target hardware backend.
        
        Parameters
        ----------
        *tensors : np.ndarray
            Arrays to cast.
        gpu_id : int, optional
            GPU ID to use if computational_method of the network is "GPU_CUDA".
        
        Returns
        -------
        :type:`~HeteroSymNN.types.BackendArray` or tuple[:type:`~HeteroSymNN.types.BackendArray`]
            The casted arrays using the internal computational manager of the network.
        """
        if (gpu_id == None):
            gpu_id = self._GPU_ID
        else:
            _validate_gpu_id(gpu_id)

        if self._COMPUTATIONAL_METHOD == "GPU_CUDA":
            with HW.be.cuda.Device(gpu_id):
                result = tuple(self._CALCULATION_MANAGER.array(t, dtype=self._DEFAULT_FLOAT_TYPE) for t in tensors)
        else:
            result = tuple(self._CALCULATION_MANAGER.array(t, dtype=self._DEFAULT_FLOAT_TYPE) for t in tensors)
            

        return result[0] if len(result) == 1 else result
    
    def asnumpy(self, *tensors: BackendArray)->Union[np.ndarray,tuple[np.ndarray]]:
        """
        Method to bring one or more backend arrays (e.g., CuPy) back to Host RAM as standard NumPy arrays.
        
        Parameters
        ----------
        *tensors : :type:`~HeteroSymNN.types.BackendArray`
            Arrays to cast

        Returns
        -------
        np.ndarray or tuple[np.ndarray]
            The `np.ndarray` version of the arrays.
        """
        result = tuple(self._ASNUMPY(t) for t in tensors)
        
        return result[0] if len(result) == 1 else result
        

    def _to(self, location:Literal["host","device"]):
        """
        Internal method to move the network parameters to the specified logical location.
        
        Parameters
        ----------
        location : Literal["host","device"]
            Location to move the network parameters to.
        """
        location = location.lower()
        self._UPDATE_METHOD._to_device(location)
        for layer in self._LAYERS:
            layer.to(location)

    def forward(self,input_values:BackendArray)->BackendArray:
        """
        Internal method to perform a forward pass through the network.
        
        Parameters
        ----------
        input_values : :type:`~HeteroSymNN.types.BackendArray`
            Input values for the network.
        
        Returns
        -------
        :type:`~HeteroSymNN.types.BackendArray`
            Output values from the network.
        """
        current_a = input_values
        for layer in self._LAYERS:
            current_a = layer.forward(current_a)

        return current_a

    def backward(self,error_values:BackendArray)->BackendArray:
        """
        Internal method to perform a backward pass through the network.
        
        Parameters
        ----------
        error_values : :type:`~HeteroSymNN.types.BackendArray`
            Error values to propagate back through the network.

        Returns
        -------
        :type:`~HeteroSymNN.types.BackendArray`
            Error values propagated back to the input layer.
        """
        self.to("device")
        next_layer_error_sum =self._CALCULATION_MANAGER.array(error_values, dtype=self._CALCULATION_MANAGER.float32)
        
        for layer in reversed(self._LAYERS):
            next_layer_error_sum = layer.backward(next_layer_error_sum)
        self._input_gradient = next_layer_error_sum
        return next_layer_error_sum
    
    def train_step(self, x_input: BackendArray, y_target: BackendArray)->float:
        """
        Perform a single training step (forward pass, loss computation, backward pass, and parameter update).

        Parameters
        ----------
        x_input : :type:`~HeteroSymNN.types.BackendArray`
            Input values to train the network on. Shape should be (batch_size, num_inputs).
        y_target : :type:`~HeteroSymNN.types.BackendArray`
            Target output values for the network. Shape should be (batch_size, num_outputs).
        
        Returns
        -------
        float
            Computed loss for the training step.
        
        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.ShapeMismatchError`
            If the input features or target features mismatch the network architecture.
        """
        self.to("device")
        
        x_input_t = x_input.T
        y_target_t = y_target.T
        
        expected_inputs = self._LAYERS[0].num_inputs
        expected_outputs = self._LAYERS[-1].num_nodes
        
        if x_input_t.shape[0] != expected_inputs:
            raise ShapeMismatchError(f"Input features mismatch. Expected {expected_inputs}, got {x_input_t.shape[0]}.")
        if y_target_t.shape[0] != expected_outputs:
            raise ShapeMismatchError(f"Target features mismatch. Expected {expected_outputs}, got {y_target_t.shape[0]}.")
        if x_input_t.shape[1] != y_target_t.shape[1]:
            raise ShapeMismatchError(f"Batch size mismatch between inputs ({x_input_t.shape[1]}) and targets ({y_target_t.shape[1]}).")

        self.num_completed_train_iterations += 1
        
        y_pred = self.forward(x_input_t)
        
        loss = self._LOSS_FUNCTION.forward(y_pred, y_target_t)
        error_to_propagate = self._LOSS_FUNCTION.backward(y_pred, y_target_t)

        self.backward(error_to_propagate)
        self.update_params()

        return self._ASNUMPY(loss)
    
    def update_params(self)->None:
        """
        Update the network parameters using the optimizer.
        """
        self.to("device")
        self._UPDATE_METHOD.step(self._LAYERS) 


    
    @clean_traceback
    def predict(self,input_values:Union[list,list[list]],to_cpu:bool = True)->Union[np.ndarray,BackendArray]:
        """
        Make predictions using the neural network.
        
        Parameters
        ----------
        input_values : list or list[list]
            Input values for making predictions. In case of multiple samples, shape should be (num_samples, num_features) or (num_features, num_samples).
        to_cpu : bool, optional
            Whether to return the predictions as a NumPy array on the CPU. If False, returns in the current :type:`~HeteroSymNN.types.BackendArray` format. by default True
        
        Returns
        -------
        np.ndarray or :type:`~HeteroSymNN.types.BackendArray`
            Predicted output values.

        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.ShapeMismatchError`
            If the input features mismatch the network architecture.
        """
        self.to("device")
        if not isinstance(input_values, (np.ndarray, self._CALCULATION_MANAGER.ndarray)):
            current_a = self._CALCULATION_MANAGER.array(input_values, dtype=self._DEFAULT_FLOAT_TYPE)
        else:
            current_a = self._CALCULATION_MANAGER.asarray(input_values, dtype=self._DEFAULT_FLOAT_TYPE)
            
        if current_a.ndim == 1:
            current_a = current_a.reshape(-1, 1)
        

        if current_a.shape[0] == self._LAYERS[0].num_inputs:
            pass
        elif current_a.shape[1] == self._LAYERS[0].num_inputs:
            current_a = current_a.T
        else:  
            raise ShapeMismatchError(f"Shape of the input is incorrect {current_a.shape}. Were expecting {self._LAYERS[0].num_inputs} features, but none dimention matched.")

        prediction = self.forward(current_a)
        if(to_cpu):
            return self._ASNUMPY(prediction).T
        
        return prediction
    
    @clean_traceback
    def get_parameters(self)->dict[Union[str,int],dict[str,np.ndarray]]:
        """
        Get the parameters (weights and biases) of the network.
        
        Returns
        -------
        dict[str,dict[str,np.ndarray]]
            Dictionary containing the parameters of each layer.
        """
        return {f'layer_{i}': layer.get_parameters() for i, layer in enumerate(self._LAYERS)}

    @property
    def total_parameters(self)->int:
        """
        Get the total number of trainable parameters in the network.

        Counts weights and biases across all layers, excluding structural
        arrays such as connection masks.

        Returns
        -------
        int
            Total number of trainable parameters.
        """
        _EXCLUDED_KEYS = {'connection_mask'}
        count = 0
        for layer in self._LAYERS:
            for key, value in layer.get_parameters().items():
                if key not in _EXCLUDED_KEYS:
                    count += value.size
        return count

    @clean_traceback
    def set_parameters(self, params:dict[Union[str,int],dict[str,np.ndarray]])->None:
        """
        Set the parameters of the network.
        
        Parameters
        ----------
        params : dict[Union[str,int],dict[str,np.ndarray]]
            Dictionary containing the parameters for each layer.
        """
        self.to("host")
        for key in params:
            index = key
            if (type(key) != int):
                index = int(key.split("_")[-1])
            self._LAYERS[index].set_parameters(params[key])

    @clean_traceback
    def change_constants(self,new_constants:dict[int,Union[list[ConstantToUpdate],ConstantToUpdate]])->None:
        """
        Change the constants in the activation functions of the network layers.
        
        Parameters
        ----------
        new_constants : dict[int,Union[list[:type:`~HeteroSymNN.types.ConstantToUpdate`], :type:`~HeteroSymNN.types.ConstantToUpdate`]]
            Dictionary mapping layer indices to new constant values for the activation functions."""
        for num_layer in new_constants.keys():
            self._LAYERS[num_layer].change_constant(new_constants[num_layer])
            
    def get_config(self)->dict[str,Any]:
        """
        Get the configuration of the neural network.
        
        Returns
        -------
        dict[str,Any]
            Dictionary containing the configuration of the network.

            Parameters include:
            
                * **"network_structure"** (*list[int]*): List of number of nodes per layer and the type of layer.
                * **"layer_configs"** (*dict[str,any]*): Configuration of each layer.
                * **"learning_rate"** (*float*): Learning rate of the network.
                * **"batch_size"** (*int*): Batch size used during training.
                * **"loss_config"** (*dict[str, Any]*): Configuration of the loss function.
                * **"num_training_epochs"** (*int*): Number of training iterations (epochs).
        """
        self.to("host")
        layers_configs = {}
        for i,layer in enumerate(self._LAYERS):
            layers_configs.update({"layer_"+str(i):layer.get_config()})

        config = {
            'network_structure': self._NETWORK_STRUCTURE,
            'layer_configs': layers_configs,
            'batch_size': self._BATCH_SIZE
        }

        config["loss_config"] = self._LOSS_FUNCTION.get_config()
        config['num_training_epochs'] = self.num_training_epochs
        return config

    def save_model(self,path: str, model_name: str = None, description: str = None,overwrite: bool = False,save_optimizer:bool = True)->str:
        """
        Saves the model architecture, parameters, optimizer state, and wrapper configuration to a file.

        The file is saved as a compressed symnn archive (``.symnn``).

        Parameters
        ----------
        path : str
            Directory path to save the file.
        model_name : str
            Name of the model that is going to be saved (will be used for the filename).
        description : str, optional
            Optional description to store in metadata.
        overwrite : bool, optional
            Whether to overwrite the file if it already exists.
        save_optimizer : bool, optional
            Whether to save the optimizer state.
            
        Returns
        -------
        str
            Full path to the saved model file.
        """
        if (model_name is None):
            model_name = self.__class__.__name__
        
        if  (path.startswith("/")):
            path = os.path.join(os.getcwd(), path)
        
        is_directory = os.path.isdir(path) or path.endswith("/") or path.endswith(os.sep)

        if is_directory:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            final_filename = f"{model_name}_{timestamp}.symnn"
            
            full_path = os.path.join(path, final_filename)
            

        else:
            if not path.endswith(".symnn"):
                full_path = path + ".symnn"
            else:
                full_path = path

            # 3. The Overwrite Guardrail
            if (os.path.exists(full_path)):
                if not(overwrite):
                    temp = full_path[:-6]
                    offset = 1
                    while (os.path.exists(temp + f"_{offset}.symnn")):
                        offset += 1
                    full_path = temp + f"_{offset}.symnn"

            full_path = Path(full_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            config_to_save,flat_params = self._get_save_objects(model_name,description,save_optimizer)

            npz_ram_buffer = io.BytesIO()
            np.savez_compressed(npz_ram_buffer, **flat_params)

            from ...utility import _NumpyEncoder

            with zipfile.ZipFile(full_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                
                archive.writestr("config.json", json.dumps(config_to_save, indent=4, cls=_NumpyEncoder))
                archive.writestr("weights.npz", npz_ram_buffer.getvalue())

            return full_path
        except Exception as e:
            raise SavingError(f"Error ocured when saving the model {model_name}. {str(e)}")

    def _get_save_objects(self,model_name: str = None, description: str = None,save_optimizer:bool = True)->tuple[dict[str,Any],dict[str,Any]]:
        """Internal method to format the data for saving."""
        self.to("host")
        architecture_config = self.get_config()
        architecture_config.pop("network_structure")

        metadata = {
            'model_name': model_name,
            'model_class': str(self.__class__.__name__),
            'description': description,
            'save_timestamp': datetime.datetime.now().isoformat(),
            'framework_version':__version__,
            'total_training_iterations': self.num_completed_train_iterations,
            'total_epochs_iterations':self.num_completed_epochs
        }

        from ...API import registries
        if (metadata["model_class"] in registries.registry.legacy_map.keys()):
            metadata["model_class"] = registries.registry.legacy_map[metadata["model_class"]]


        
        config_to_save = {
            'architecture': architecture_config,
            'metadata': metadata
        }

        if (save_optimizer):
            config_to_save['optimizer_config'] = self._UPDATE_METHOD.get_config()
        

        params = self.get_parameters()

        flat_params = {}
        for layer_key, layer_params in params.items():
            for param_key, param_value in layer_params.items():
                flat_params[f"{layer_key}_{param_key}"] = param_value
        
        if (save_optimizer):
            opt_global, opt_layers = self._UPDATE_METHOD.get_state()

            for param_key, val in opt_global.items():
                flat_params[f"global_opt_{param_key}"] = val 

            for i, layer_opt_dict in enumerate(opt_layers):
                for param_key, matrix in layer_opt_dict.items():
                    flat_params[f"layer_{i}_opt_{param_key}"] = matrix
        
        return config_to_save,flat_params

    @classmethod
    def _extract_symnn_archive(cls, path: str) -> tuple["BaseNetwork", dict[str, Any]]:
        """
        Internal method to load a .symnn archive.
        
        Parameters
        ----------
        path : str
            Path to the .symnn archive.
        """
        if not path.endswith(".symnn"):
            path = path + ".symnn"

        if not os.path.exists(path):
            raise PathError(f"No file found: {path}")

        try:
            from ...API import registries
            with zipfile.ZipFile(path, 'r') as archive:
                config_bytes = archive.read("config.json")
                config_wrapper = json.loads(config_bytes)
                
                metadata = config_wrapper.get('metadata', {})
                
                model_class_name = metadata.get('model_class')
                TargetNetworkClass = registries.registry._net_map[model_class_name]
                
                model = TargetNetworkClass.from_config(config_wrapper, registry_module=registries.registry)

                npz_bytes = archive.read("weights.npz")
                npz_ram_buffer = io.BytesIO(npz_bytes)
                
                layer_params_dict = {}
                opt_global = {}
                opt_layers_dicts = [None]*len(model.layers)
                
                with np.load(npz_ram_buffer, allow_pickle=False) as data:
                    for key, value in data.items():
                        if key.startswith("global_opt_"):
                            param_name = key.replace("global_opt_", "", 1)
                            opt_global[param_name] = value.item() if value.ndim == 0 else value

                        elif "_opt_" in key:
                            parts = key.split("_opt_", 1)
                            layer_idx = int(parts[0].split("_")[1])
                            param_name = parts[1]                   
                            
                            if (opt_layers_dicts[layer_idx] is None):
                                opt_layers_dicts[layer_idx] = {}
                            opt_layers_dicts[layer_idx][param_name] = value
                            
                        elif key.startswith("layer_"):
                            parts = key.split("_", 2)
                            layer_key = f"{parts[0]}_{parts[1]}"
                            param_key = parts[2]                 
                            
                            if layer_key not in layer_params_dict:
                                layer_params_dict[layer_key] = {}
                            layer_params_dict[layer_key][param_key] = value
                
                model.set_parameters(layer_params_dict)

                if all(x is None for x in opt_layers_dicts):
                    opt_layers_dicts = []
                else:
                    opt_layers_dicts = [x if x is not None else {} for x in opt_layers_dicts]

                rebuilt_opt_state = {"layer_states":opt_layers_dicts}
                rebuilt_opt_state.update(opt_global)

                model._UPDATE_METHOD.set_state(rebuilt_opt_state, model._CALCULATION_MANAGER)
                model._UPDATE_METHOD._initialize_state(model.layers)

                model.num_completed_train_iterations = metadata.get('total_training_iterations', 0)
                model.num_completed_epochs = metadata.get('total_epochs_iterations', 0)

                return model, metadata
        except Exception as e:
            raise LoadingError(f"Failed to load the model from {path}. The .symnn archive may be corrupted. Cause: {e}") from e

    @classmethod
    @clean_traceback
    def load_model(cls, path: str) -> "BaseNetwork":
        """
        Creates a brand new BaseNetwork (or subclass) and populates it directly from a .symnn archive.
        
        Parameters
        ----------
        path : str
            Path to the .symnn archive.
        """
        model, _ = cls._extract_symnn_archive(path)
        return model

    @clean_traceback
    def load_state(self, path: str) -> None:
        """
        Loads a model from a ``.symnn`` ZIP archive created by :meth:`save_model` and overwrites the current parameters of the model.

        Reconstructs the Network weights, optimizer state, etc.

        Parameters
        ----------
        path : str
            Path to the ``.symnn`` file.
        
        Raises
        ------
        IOError
            If the file cannot be loaded or has an invalid format.
        """
        model, metadata = self._extract_symnn_archive(path)
        
        if len(self.layers) != len(model.layers):
            raise LoadingError("Cannot load state: the saved model has a different number of layers.")
            
        for i, (l_self, l_model) in enumerate(zip(self.layers, model.layers)):
            if l_self.__class__ != l_model.__class__:
                raise LoadingError(f"Cannot load state: layer {i} type mismatch ({l_self.__class__.__name__} vs {l_model.__class__.__name__}).")
            if l_self.num_inputs != l_model.num_inputs or l_self.num_nodes != l_model.num_nodes:
                raise LoadingError(f"Cannot load state: layer {i} dimensions mismatch.")
                
        self.set_parameters(model.get_parameters())
        
        opt_global, opt_layers = model._UPDATE_METHOD.get_state()
        rebuilt_opt_state = {"layer_states": opt_layers}
        rebuilt_opt_state.update(opt_global)
        
        self._UPDATE_METHOD.set_state(rebuilt_opt_state, self._CALCULATION_MANAGER)
        self._UPDATE_METHOD._initialize_state(self.layers)
        
        self.num_completed_train_iterations = model.num_completed_train_iterations
        self.num_completed_epochs = model.num_completed_epochs