from __future__ import annotations
import numpy as np
from typing import Literal, Union, Sequence
import warnings

from ..Backend import hardware as HW
from ..Backend.validators import _validate_gpu_id
from ..types import LayerConstruction,NodeConfig,BackendArray,ConstantToUpdate
from ..JIT.compiler import SymbolicJITCompiler
from .initializers import Initializer
from ..config import settings
from ..exceptions import LayerConfigurationError, RuntimeStateError, BackendNotAvailableWarning, HardwareWarning,JITError, ComputationalMethodValueError, HeteroSymNNValueError
from ..error_handlers import clean_traceback

__all__ = ["BaseLayer", "LinearLayer"]

class BaseLayer:
    """
        Base layer class for all layer types.
        
        Parameters
        ----------
        input_shape : int
            Number of inputs the layer is going to receive.
        layer_configuration : :type:`~HeteroSymNN.types.LayerConstruction`
            Configuration of the layer including the node activation functions and their constants as well as the initializer used for the parameters of the layer.
        batch_size : int, optional
            Initial batch size for the layer, by default 1.
        gpu_id : int, optional
            Id of the GPU to use if available, by default 0.

        Attributes
        ----------
        delta: :type:`HeteroSymnn.types.BackendArray`
            values per neuron of the contribution to the error of the layer.
        a: :type:`HeteroSymnn.types.BackendArray`
            values per neuron of the output of the layer.
        z:  :type:`HeteroSymnn.types.BackendArray`
            values per neuron of the input of the layer after applying the mask and the biases.
    """
    @clean_traceback
    def __init__(self,input_shape:int,layer_configuration:LayerConstruction,batch_size:int = 1,gpu_id:int = 0):
        
        self._CALCULATION_MANAGER = settings.default_manager
        self._ASNUMPY = settings.default_asnumpy
        _validate_gpu_id(gpu_id)
        self._GPU_ID = gpu_id
        self._CURRENT_DEVICE = "CPU"
        self._CURRENT_LOCATION = "host"
        self._COMPUTATIONAL_METHOD = settings.default_compute_method
        self._CURRENT_VECTOR_FORMAT = self._ASNUMPY
        self._DEFAULT_FLOAT_TYPE = settings.default_dtype

        self._num_inputs = input_shape
        self._num_nodes = len(layer_configuration[0])
        self._layer_node_configs = layer_configuration[0]
        self._initializer = layer_configuration[1]
        self.delta = None
        self.a = None
        self.z = None

        if (self._COMPUTATIONAL_METHOD.split("_")[0] == "GPU"):
            with HW.be.cuda.Device(self._GPU_ID):
                self._funcs_constats,self.param_offsets = self._generate_constant_array(layer_configuration[0])
        else:
            self._funcs_constats,self.param_offsets = self._generate_constant_array(layer_configuration[0])

        self.batch_size_change(batch_size)
        try:
            self._act_funcions_manager = SymbolicJITCompiler(layer_configuration[0],self._COMPUTATIONAL_METHOD,self._GPU_ID)
        except JITError as e:
            raise LayerConfigurationError(f"Error in layer setup due to a problem during kernel creation: {str(e)}")

    @property
    def initializer(self)->Initializer:
        """
        Iinititalizer used in the creation of the layer weights, biases and connection mask.
        
        Returns
        -------
        :class:`~HeteroSymNN.Core.Nets.initializers.Initializer`
        """
        return self._initializer

    @property
    def num_nodes(self)->int:
        """
        Property to get the number of nodes in the layer.

        Returns
        -------
        int
            Number of nodes in the layer.
        """
        return self._num_nodes

    @property
    def num_inputs(self)->int:
        """
        Property to get the number of inputs to the layer.

        Returns
        -------
        int
            Number of inputs to the layer.
        """
        return self._num_inputs

    @property
    def gpu_id(self)->int:
        """
        Property to get the current GPU ID being used for calculations.

        If want to set a new GPU ID use :meth:`set_gpu_id`.

        Returns
        -------
        int
            Current GPU ID.
        """
        return self._GPU_ID
    
    @property
    def computational_method(self)->Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]:
        """
        Property to get the current computational method being used for calculations.

        Returns
        -------
        Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]
            Used computational method.
        """
        return self._COMPUTATIONAL_METHOD
    
    @property
    def activation_function_constants(self)->BackendArray:
        """
        Property to get the current activation function constants array.

        To change any constant or list of constants use :meth:`change_constant`.

        Returns
        -------
        :type:`~HeteroSymNN.types.BackendArray`
            Array of activation function constants.
        """
        return self._funcs_constats
    
    @property
    def current_device(self)->Literal["CPU","GPU"]:
        """
        Property to get the current device where the parameters are located.

        Returns
        -------
        Literal["CPU","GPU"]
            Location of the layer parameters.
        """
        return self._CURRENT_DEVICE
    
    @property
    def current_location(self)->Literal["host","device"]:
        """
        Property to get the current logical location where the parameters are located.

        Returns
        -------
        Literal["host","device"]
            Logical location of the layer parameters.
        """
        return self._CURRENT_LOCATION
    
    @property
    def initial_nodes_layer_configs(self)->Sequence[NodeConfig]:
        """
        Property to get the initial layer node configurations used during layer construction.

        Returns
        -------
        list[:type:`~HeteroSymNN.types.NodeConfig`]
            List of :type:`~HeteroSymNN.types.NodeConfig` used during layer construction.
        """
        return self._layer_node_configs
    
    def reset_parameters(self,reset_constants:bool = False)->None:
        """
        Method to reset the parameters of the layer to their initial values.

        Parameters
        ----------
        reset_constants: bool, optional
            Bool value if you want to also reset the activation function constants, by default False.
        """
        self.to("host")
        if (reset_constants):
            if (self._COMPUTATIONAL_METHOD.split("_")[0] == "GPU"):
                with HW.be.cuda.Device(self._GPU_ID):
                    self._funcs_constats,self.param_offsets = self._generate_constant_array(self._layer_node_configs)
            else:
                self._funcs_constats,self.param_offsets = self._generate_constant_array(self._layer_node_configs)

    def _generate_constant_array(self,activation_functions:list[NodeConfig])->tuple[np.ndarray,BackendArray]:
        """
        Internal function for generating the array for the constants used in the activation functions.

        Parameters
        ---------- 
        activation_functions : list[:type:`~HeteroSymNN.types.NodeConfig`]
            List of :type:`~HeteroSymNN.types.NodeConfig` for the layer.

        Returns
        -------
        tuple[np.ndarray, :type:`~HeteroSymNN.types.BackendArray`]
            Tuple containing the array of constants and the offsets for each node.
        """
        temp = []
        offsets = []
        constant_counter = 0
        self.CONSTANT_DICT:dict[int,dict[str,int]] = {}
        for i,config in enumerate(activation_functions):
            sorted_keys = sorted(config[1].keys())
            offsets.append(constant_counter)
            node_constant_dict = {}
            for constant in sorted_keys:
                node_constant_dict.update({constant:constant_counter})
                constant_counter += 1
                temp.append(config[1][constant])
            self.CONSTANT_DICT.update({i:node_constant_dict})
        
        if (len(temp)==0):
            return np.array([0.0],dtype=self._DEFAULT_FLOAT_TYPE),self._CALCULATION_MANAGER.array([0]*self._num_nodes)
        return np.array(temp,dtype=self._DEFAULT_FLOAT_TYPE),self._CALCULATION_MANAGER.array(offsets)

    def reconstruct_layer_config (self)->Sequence[NodeConfig]:
        """
        Regenerates the list of :type:`~HeteroSymNN.types.NodeConfig` with the updated activation funcion constants.

        Returns
        -------
        list[:type:`~HeteroSymNN.types.NodeConfig`]
            List of :type:`~HeteroSymNN.types.NodeConfig` with the current constants.
        """
        self.to("host")
        new_layer_config = []
        for i,node in enumerate(self._layer_node_configs):
            node_constant_dict = {}
            sorted_constants = sorted(node[1].keys())
            for key in sorted_constants:
                value = self._funcs_constats[self.CONSTANT_DICT[i][key]]
                node_constant_dict[key] = float(value)

            new_layer_config.append((node[0],node_constant_dict))
        return new_layer_config
    
    def set_gpu_id(self,new_id:int)->None:
        """
        Method to change the GPU that is going to be used for the calculations.
        
        Parameters
        ----------
        new_id : int
            New GPU Id to use.

        :exc:`~HeteroSymNN.exceptions.InvalidDeviceIDError`
            If the GPU ID is negative or greater than or equal to the number of available GPUs.
        """
        if (new_id != self._GPU_ID):
            _validate_gpu_id(new_id)
            self._GPU_ID = new_id
            self._act_funcions_manager.set_gpu_id(self._GPU_ID)
    
    @clean_traceback
    def _change_COMPUTATIONAL_METHOD(self,new_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],gpu_id:int = None)->Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]:
        """
        Method to force the layer to use a specific computational method. *WARNING*, this will forece a kernel recompilation.
        
        Parameters
        ----------
        new_method : Literal["GPU_CUDA", "CPU_JIT", "CPU_PYTHON"]
            String of the new computational method.
        gpu_id : int, optional
            When converting from a CPU method to the GPU method can set a GPU Id of not pass it utilize the last set GPU Id.

        Returns
        -------
        Literal["GPU_CUDA", "CPU_JIT", "CPU_PYTHON"]
            The computational method that was set. This could be different from the requested one if the requested one is not available or encontered an error with out :attr:`~HeteroSymNN.config.settings.warning_level` been set to "error".
        
        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.ComputationalMethodValueError`
            If the passed value is not "GPU_CUDA", "CPU_JIT" or "CPU_PYTHON".
        """
        new_method = new_method.upper()
        try_method = new_method
        msg_extra = ""
        if not(new_method in ["GPU_CUDA","CPU_JIT","CPU_PYTHON"]):
            raise ComputationalMethodValueError("tried to change the computational method to something that isn't GPU_CUDA, CPU_JIT or CPU_PYTHON")
        
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
            self._COMPUTATIONAL_METHOD = self._act_funcions_manager._change_method(new_method,self._GPU_ID)
            self.param_offsets = self._ASNUMPY(self.param_offsets)
            self._GPU_ID = gpu_id
            batch_size = self.z.shape[1]
            self.z = None
            self.a = None
            self.delta = None
            if ("CPU" in self._COMPUTATIONAL_METHOD):
                self._CALCULATION_MANAGER = np
                self._ASNUMPY = np.array
                self.param_offsets = self._CALCULATION_MANAGER.array(self.param_offsets)
            elif ("GPU" in self._COMPUTATIONAL_METHOD):
                self._CALCULATION_MANAGER = HW.cp
                self._ASNUMPY = HW.cp.asnumpy
                with HW.be.cuda.Device(self._GPU_ID):
                    self.param_offsets = self._CALCULATION_MANAGER.array(self.param_offsets)

            self.batch_size_change(batch_size)
        return self._COMPUTATIONAL_METHOD

    def to(self,location:Literal["host","device"])->None:
        """
        Change the logical location of the weights, biases, connection mask and activation function constants to host or device.

        Parameters
        ----------
        location : Literal["host", "device"]
            To which logical location to change the data of the layer.
        
        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.HeteroSymNNValueError`
            If the passed value is not "host" or "device".
        """
        location = location.lower()
        if not(location in ["host","device"]):
            raise HeteroSymNNValueError("Location not recognized. Expecting host or device.")
        
        target_hardware = "CPU"
        if location == "device":
             if "GPU" in self._COMPUTATIONAL_METHOD:
                 target_hardware = "GPU"
        
        new_vector_format = self._CALCULATION_MANAGER.array
        if target_hardware == "CPU":
            new_vector_format = self._ASNUMPY

        if(location != self._CURRENT_LOCATION):
            self._CURRENT_LOCATION = location
            self._CURRENT_DEVICE = target_hardware
            self._CURRENT_VECTOR_FORMAT = new_vector_format
            if (target_hardware == "GPU"):
                with HW.be.cuda.Device(self._GPU_ID):
                    self._funcs_constats = self._CURRENT_VECTOR_FORMAT(self._funcs_constats)
            else:
                self._funcs_constats = self._CURRENT_VECTOR_FORMAT(self._funcs_constats)

    def batch_size_change(self, batch_size: int)->None:
        """
        Method to change the batch size of the layer. This will reallocate the internal buffers if needed.

        Parameters
        ----------
        batch_size : int
            New batch size for the layer.
        """
        expected_shape = (self._num_nodes, batch_size)

        if  ((self.z is None)or(self.z.shape != expected_shape)):
            if ("GPU" in self._COMPUTATIONAL_METHOD):
                with HW.be.cuda.Device(self._GPU_ID):
                    self.z = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
                    self.a = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
                    self.delta = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
            else:
                self.z = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
                self.a = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
                self.delta = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)

    @property
    def working_parameters(self) -> dict[str, BackendArray]:
        """Returns the live, mutable parameter arrays (e.g., weights, biases) currently stored on the active computational device.

            These are the actual arrays used and updated during training. The implementation details are the responsibility of the subclasses.

            Return
            ------
            dict[str, :type:`~HeteroSymNN.types.BackendArray`]
                Dictionary mapping parameter names to their current :type:`~HeteroSymNN.types.BackendArray` objects.
        """
        return {}

    @property
    def working_gradients(self) -> dict[str, BackendArray]:
        """Returns the live gradient arrays accumulated during the backward pass, stored on the active computational device.

            These gradients are directly consumed by Optimizers to update the working parameters. The implementation details are the responsibility of the subclasses.

            Return
            ------
            dict[str, :type:`~HeteroSymNN.types.BackendArray`]
                Dictionary mapping parameter names to their corresponding gradient :type:`~HeteroSymNN.types.BackendArray` objects.
        """
        return {}
        
    @property
    def working_masks(self) -> dict[str, BackendArray]:
        """Returns the live sparsity or structural masks on the active computational device.

            These masks define fixed constraints (like inactive connections) that multiply the parameters during forward passes. The implementation details are the responsibility of the subclasses.

            Return
            ------
            dict[str, :type:`~HeteroSymNN.types.BackendArray`]
                Dictionary mapping parameter names to their mask :type:`~HeteroSymNN.types.BackendArray` objects.
        """
        return {}

    def forward(self,input_values:BackendArray)->BackendArray:
        """
        Method to perform the forward pass of the layer.
        
        Parameters
        ----------
        input_values : :type:`~HeteroSymNN.types.BackendArray`
            Input values to the layer.

        Returns
        -------
        :type:`~HeteroSymNN.types.BackendArray`
            Output values of the layer after applying the activation functions.
        """
        raise NotImplementedError

    def backward(self,error_values:BackendArray)->BackendArray:
        """
        Method to perform the backward pass of the layer.
        
        Parameters
        ----------
        error_values : :type:`~HeteroSymNN.types.BackendArray`
            Error values from the next layer.

        Returns
        ------- 
        :type:`~HeteroSymNN.types.BackendArray`
            Error values to be passed to the previous layer.
        """
        raise NotImplementedError
    
    def change_constant(self,new_values:Union[list[ConstantToUpdate],ConstantToUpdate])-> None:
        """
        Method to change the value of a activation function constant of a node or list of nodes.
        
        Parameters
        ----------
        new_values: Union[list[:type:`~HeteroSymNN.types.ConstantToUpdate`], :type:`~HeteroSymNN.types.ConstantToUpdate`]
            New value or list of new values to set. Each value is a tuple containing the node index, the constant name, and the new value.
        """
        if (isinstance(new_values[0], (int, np.integer))):
            new_values = [new_values]
        
        for value in new_values:
            traductor = self.CONSTANT_DICT[value[0]]
            self._funcs_constats[traductor[value[1]]] = value[2]

    def get_parameters(self)->dict[str,np.ndarray]:
        """
        Method to get the parameters of the layer.
        
        This method must be implemented by the subclasses.

        Returns
        ------
        dict[str, ndarray]:
            Dictionary of the parameters by string. 
        """
        raise NotImplementedError
        
    def set_parameters(self, params:dict[str,np.ndarray])->None:
        """
        Method to set the parameters of the layer.
        
        This method must be implemented by the subclasses.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Dictionary containing the new parameters with string keys.
        """
        raise NotImplementedError
    
    def get_config(self)->dict[str,any]:
        """
        Method to get the configuration of the layer.
        
        This method must be implemented by the subclasses.

        Returns
        -------
        dict[str, any]
            Dictionary containing the configuration of the layer.
        """
        return  {"num_nodes":self._num_nodes,
                "num_inputs":self._num_inputs,
                "layer_node_configs":self.reconstruct_layer_config (),
                "initializer":self._initializer.get_config(),
                "layer_type":self.__class__.__name__
                }

class LinearLayer(BaseLayer):
    """
        Linear layer class
        
        Parameters
        ----------
        num_inputs : int
            Number of inputs the layer is going to receive.
        layer_configuration : :type:`~HeteroSymNN.types.LayerConstructionConfig`
            Configuration of the layer including the node activation functions and their constants as well as the initializer to set the weights, biases and connection mask.
        batch_size : int, optional
            Initial batch size for the layer, by default 1.
        Gpu_id : int, optional
            Id of the GPU to use if available, by default 0.3
        
        Attributes
        ----------
        delta: :type:`HeteroSymnn.types.BackendArray`
            values per neuron of the contribution to the error of the layer.
        a: :type:`HeteroSymnn.types.BackendArray`
            values per neuron of the output of the layer.
        z:  :type:`HeteroSymnn.types.BackendArray`
            values per neuron of the input of the layer after applying the mask and the biases.
        
        Examples
        --------
        Generaly one doesn't need to instanciate this class directly but if some want to do it, here is an example of how to do it.

        >>> from HeteroSymNN.Core.layers import LinearLayer
        >>> from HeteroSymNN.Core import initializers
        >>> custom_initializer = initializers.HeNormal()
        >>> layer_config = ([("relu", {}),("relu", {}),("sigmoid", {}),("relu", {}),("relu", {})],custom_initializer)
        >>> layer = LinearLayer(
        ...    num_inputs=3,
        ...    layer_configuration=layer_config
        ... )
        
        
    """
    def __init__(self,num_inputs:int,layer_configuration:LayerConstruction,batch_size:int = 1,Gpu_id:int = 0):
        super().__init__(num_inputs,layer_configuration,batch_size,Gpu_id)

        init_biases = self._initializer.generate_constant([self._num_nodes,1])
        init_weights = self._initializer.generate_from_distribution([self._num_nodes,self._num_inputs],self._num_inputs,self._num_nodes)
        init_mask = self._initializer.generate_binary_mask([self._num_nodes,self._num_inputs])
        self._biases = np.array(init_biases).astype(self._DEFAULT_FLOAT_TYPE)
        self._weights = np.array(init_weights).astype(self._DEFAULT_FLOAT_TYPE)
        self._connection_mask = np.array(init_mask).astype(self._DEFAULT_FLOAT_TYPE)
        self._grad_weights = np.zeros_like(self._weights,dtype=self._DEFAULT_FLOAT_TYPE)
        self._grad_biases = np.zeros_like(self._biases,dtype=self._DEFAULT_FLOAT_TYPE)
        self._cached_input:BackendArray = None

    @property
    def biases(self)->np.ndarray:
        """
        Property to get the biases of the layer.

        Returns
        -------
        np.ndarray
            Biases of the layer.
        """
        return self._ASNUMPY(self._biases)
    
    @property
    def weights(self)->np.ndarray:
        """
        Property to get the weights of the layer.

        Returns
        -------
        np.ndarray
            Weights of the layer.
        """
        return self._ASNUMPY(self._weights*self._connection_mask)
    
    @property
    def working_parameters(self) -> dict[str, BackendArray]:
        """Returns the live, mutable parameter arrays currently stored on the active computational device.

            Return
            ------
            dict[str, :type:`~HeteroSymNN.types.BackendArray`]
                Dictionary mapping parameter names to their current :type:`~HeteroSymNN.types.BackendArray` objects.

                "weights": :type:`~HeteroSymNN.types.BackendArray`
                "biases": :type:`~HeteroSymNN.types.BackendArray`
        """
        return {
            "weights": self._weights,
            "biases": self._biases
        }

    @property
    def working_gradients(self) -> dict[str, BackendArray]:
        """Returns the live gradient arrays accumulated during the backward pass, stored on the active computational device.

            Return
            ------
            dict[str, :type:`~HeteroSymNN.types.BackendArray`]
                Dictionary mapping parameter names to their corresponding gradient :type:`~HeteroSymNN.types.BackendArray` objects.

                "weights": :type:`~HeteroSymNN.types.BackendArray`
                "biases": :type:`~HeteroSymNN.types.BackendArray`
        """
        return {
            "weights": self._grad_weights,
            "biases": self._grad_biases
        }
        
    @property
    def working_masks(self) -> dict[str, BackendArray]:
        """Returns the live sparsity or structural masks on the active computational device.

            Return
            ------
            dict[str, :type:`~HeteroSymNN.types.BackendArray`]
                Dictionary mapping parameter names to their mask :type:`~HeteroSymNN.types.BackendArray` objects.

                'weights': :type:`~HeteroSymNN.types.BackendArray`
        """
        return {
            "weights": self._connection_mask
        }
    
  
    def reset_parameters(self,reset_constants:bool = False,reset_mask:bool = False)->None:
        """
        Method to reset the parameters of the layer to their initial values.

        Parameters
        ----------
        reset_constants: bool, optional
            Bool value if you want to also reset the activation function constants, by default False.
        reset_mask: bool, optional
            Bool value if you want to also reset the connection mask, by default False.
        """
        super().reset_parameters(reset_constants,reset_mask)
        init_biases = self._initializer.generate_constant([self._num_nodes,1])
        init_weights = self._initializer.generate_from_distribution([self._num_nodes,self._num_inputs],self._num_inputs,self._num_nodes)
        init_mask = self._initializer.generate_binary_mask([self._num_nodes,self._num_inputs])

        self._biases = np.array(init_biases).astype(self._DEFAULT_FLOAT_TYPE)
        self._weights = np.array(init_weights).astype(self._DEFAULT_FLOAT_TYPE)
        if (reset_mask):
            self._connection_mask = np.array(init_mask).astype(self._DEFAULT_FLOAT_TYPE)


    def to(self,location:Literal["host","device"])->None:
        """
        Change the logical location of the weights, biases, connection mask and activation function constants to host or device.

        Parameters
        ----------
        location : Literal["host", "device"]
            To which logical location to change the data of the layer.
        """
        location = location.lower()
        same = (location == self._CURRENT_LOCATION)
        super().to(location)
        if((location == self._CURRENT_LOCATION)and not (same)):
            if (self._CURRENT_DEVICE == "GPU"):
                with HW.be.cuda.Device(self._GPU_ID):
                    self._weights = self._CURRENT_VECTOR_FORMAT(self._weights)
                    self._biases = self._CURRENT_VECTOR_FORMAT(self._biases)
                    self._connection_mask = self._CURRENT_VECTOR_FORMAT(self._connection_mask)
                    self._grad_weights = self._CURRENT_VECTOR_FORMAT(self._grad_weights)
                    self._grad_biases = self._CURRENT_VECTOR_FORMAT(self._grad_biases)
            else:
                self._weights = self._CURRENT_VECTOR_FORMAT(self._weights)
                self._biases = self._CURRENT_VECTOR_FORMAT(self._biases)
                self._connection_mask = self._CURRENT_VECTOR_FORMAT(self._connection_mask)
                self._grad_weights = self._CURRENT_VECTOR_FORMAT(self._grad_weights)
                self._grad_biases = self._CURRENT_VECTOR_FORMAT(self._grad_biases)


    def forward(self,input_values:BackendArray)->BackendArray:
        """
        Method to perform the forward pass of the layer.
        
        Parameters
        ----------
        input_values : :type:`~HeteroSymNN.types.BackendArray`
            Input values to the layer.

        Returns
        -------
        :type:`~HeteroSymNN.types.BackendArray`
            Output values of the layer after applying the activation functions.
        """
        self._cached_input = input_values
        batch_size = input_values.shape[1]
        self.batch_size_change(batch_size)
        effective_weights = self._weights * self._connection_mask
        self.z = self._CALCULATION_MANAGER.dot(effective_weights, input_values) + self._biases

        self._act_funcions_manager.forward_kernel(self.z, self.a, self._funcs_constats, self.param_offsets, self._num_nodes,batch_size)
        return self.a

    def backward(self,error_values:BackendArray)->BackendArray:
        """
        Method to perform the backward pass of the layer.
        
        Parameters
        ----------
        error_values : :type:`~HeteroSymNN.types.BackendArray`
            Error values from the next layer.

        Returns
        -------
        :type:`~HeteroSymNN.types.BackendArray`
            Error values to be passed to the previous layer.
        """
        if (self._cached_input is None):
            raise RuntimeStateError("Can't calculate the backward pass if a forward pass has not been done yet.")
        batch_size = self.z.shape[1]
        self._act_funcions_manager.backward_kernel(self.z, error_values, self.delta, self._funcs_constats, self.param_offsets, self._num_nodes,batch_size)
        self._grad_biases = self._CALCULATION_MANAGER.mean(self.delta, axis=1, keepdims=True)
        self._grad_weights = self._CALCULATION_MANAGER.dot(self.delta, self._cached_input.T) / self._DEFAULT_FLOAT_TYPE(batch_size)
        effective_weights = self._weights * self._connection_mask
        prev_layer_error_sum = self._CALCULATION_MANAGER.dot(effective_weights.T, self.delta)
        
        return prev_layer_error_sum

    def get_parameters(self)->dict[str,np.ndarray]:
        """
        Method to get the parameters of the layer.
        
        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the parameters with keys 'weights', 'biases' and 'connection_mask'.
        """
        return {
            'weights': self._ASNUMPY(self._weights).T,
            'biases': self._ASNUMPY(self._biases).T,
            'connection_mask': self._ASNUMPY(self._connection_mask).T
        }
        
    def set_parameters(self, params:dict[str,np.ndarray])->None:
        """
        Method to set the parameters of the layer.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Dictionary containing the new parameters with keys 'weights', 'biases' and 'connection_mask'.
        """
        correct_weights = (params["weights"].T.shape == self._weights.shape)
        correct_biases = (params["biases"].T.shape == self._biases.shape)
        if not(correct_biases or correct_weights):
            raise LayerConfigurationError(f"""
                Weights and biases are not in the correct dimentions. Expected {self._weights.T.shape} for the weights and {self._biases.T.shape} for the biases.
                Received {params["weights"].shape} for the weights and {params["biases"].shape} for the biases.
                 """)
        elif not(correct_weights):
            raise LayerConfigurationError(f"""
                Weights are not in the correct dimentions. Expected {self._weights.T.shape} for the weights.
                Received {params["weights"].shape} for the weights.
                """)
        elif not (correct_biases):
            raise LayerConfigurationError(f"""
                Biases are not in the correct dimentions. Expected {self._biases.T.shape} for the biases.
                Received {params["biases"].shape} for the biases.
                """)
        
        if ("connection_mask" in params.keys()):
            self.set_connection_mask(params['connection_mask'].T)
        self._weights = np.array(params['weights'].T, dtype=self._DEFAULT_FLOAT_TYPE)
        self._biases = np.array(params['biases'].T, dtype=self._DEFAULT_FLOAT_TYPE)
    
    def set_connection_mask(self, connection_mask: np.ndarray)->None:
        """
        Method to set the connection mask of the layer.

        Parameters
        ----------
        connection_mask : np.ndarray
            2D array of ones and zeros representing the connection mask. in the of shape (num_nodes, num_inputs).
        """
        if (connection_mask.shape != (self._num_nodes,self._num_inputs)):
            raise LayerConfigurationError(f"Connection mask is not in the correct dimentions. Expected ({self._num_nodes},{self._num_inputs}), received {connection_mask.shape} instead.)")

        if self._CURRENT_DEVICE == 'GPU':
            self._connection_mask = self._CALCULATION_MANAGER.array(connection_mask,dtype=self._CALCULATION_MANAGER.float32)
        else:
            self._connection_mask = connection_mask.astype(self._DEFAULT_FLOAT_TYPE)
    
    def get_config(self)->dict[str,any]:
        config = super().get_config()
        return config

class RecurrentLayer(BaseLayer):
    """
        Recurrent layer class.

        *Class not implemented yet*
    """
    def __init__(self,num_inputs:int,layer_configuration:LayerConstruction,batch_size:int = 1,Gpu_id:int = 0):
        
        raise NotImplementedError
        super().__init__(num_inputs,layer_configuration,batch_size,Gpu_id)