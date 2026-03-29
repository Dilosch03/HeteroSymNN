from __future__ import annotations
import numpy as np
from typing import Literal, Union, Sequence
import warnings

from ..Backend import hardware as HW
from ..types import LayerConstruction,NodeConfig,BackendArray,ConstantToUpdate
from ..JIT.compiler import SymbolicJITCompiler
from .initializers import Initializer
from ..config import settings
from ..exceptions import LayerConfigurationError,BackendNotAvailableError,PerformanceWarning,HardwareWarning, InvalidDeviceIDError

class BaseLayer:
    """
        Base layer class for all layer types.
        
        Parameters
        ----------
        _num_inputs : int
            Number of inputs the layer is going to receive.
        layer_configuration : :obj:`~HeteroSymNN.types.LayerConstruction`
            Configuration of the layer including the node activation functions and their constants as well as the initial _weights, _biases and connection mask.
        batch_size : int, optional
            Initial batch size for the layer, by default 1.
        gpu_id : int, optional
            Id of the GPU to use if available, by default 0.
    """
    def __init__(self,num_inputs:int,layer_configuration:LayerConstruction,batch_size:int = 1,Gpu_id:int = 0):
        
        self._CALCULATION_MANAGER = HW.be
        self._ASNUMPY = HW.asnumpy
        self._GPU_ID = Gpu_id
        self._CURRENT_DEVICE = "CPU"
        self._COMPUTATIONAL_METHOD = settings.default_compute_method
        self._CURRENT_VECTOR_FORMAT = self._ASNUMPY
        self._DEFAULT_FLOAT_TYPE = settings.default_dtype

        self._num_inputs = num_inputs
        self._num_nodes = len(layer_configuration[0])
        self._layer_node_configs = layer_configuration[0]
        self._initializer = layer_configuration[1]

        if (self._COMPUTATIONAL_METHOD.split("_")[0] == "GPU"):
            with HW.be.cuda.Device(self._GPU_ID):
                self._funcs_constats,self.param_offsets = self._generate_constant_array(layer_configuration[0])
        else:
            self._funcs_constats,self.param_offsets = self._generate_constant_array(layer_configuration[0])

        self.batch_size_change(batch_size)

        self._act_funcions_manager = SymbolicJITCompiler(layer_configuration[0],self._COMPUTATIONAL_METHOD,self._GPU_ID)

    @property
    def initializer(self)->Initializer:
        """
        Iinititalizer used in the creation of the layer weights, biases and connection mask.
        
        Returns
        -------
        :obj:`~HeteroSymNN.Core.Nets.initializers.Initializer`
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

        If want to set a new GPU ID use :obj:`set_gpu_id`.

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

        To change any constant or list of constants use :obj:`change_constant`.

        Returns
        -------
        :obj:`~HeteroSymNN.types.BackendArray`
            Array of activation function constants.
        """
        return self._funcs_constats
    
    @property
    def current_device(self)->Literal["CPU","GPU"]:
        """
        Property to get the current device where the parameters like _weights, _biases and connection mask are located.

        Returns
        -------
        Literal["CPU","GPU"]
            Location of the layer parameters.
        """
        return self._CURRENT_DEVICE
    
    @property
    def inicial_nodes_layer_configs(self)->Sequence[NodeConfig]:
        """
        Property to get the initial layer node configurations used during layer construction.

        Returns
        -------
        list[:obj:`~HeteroSymNN.types.NodeConfig`]
            List of :obj:`~HeteroSymNN.types.NodeConfig` used during layer construction.
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
        self.to("CPU")
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
        activation_functions : list[:obj:`~HeteroSymNN.types.NodeConfig`]
            List of :obj:`~HeteroSymNN.types.NodeConfig` for the layer.

        Returns
        -------
        tuple[np.ndarray, :obj:`~HeteroSymNN.types.BackendArray`]
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

    def recunstruct_layer_config(self)->Sequence[NodeConfig]:
        """
        Regenerates the list of NodeConfig with the updated activation funcion constants.

        Returns
        -------
        list[:obj:`~HeteroSymNN.types.NodeConfig`]
            List of :obj:`~HeteroSymNN.types.NodeConfig` with the current constants.
        """
        self._to("CPU")
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

        If computer does not have valid GPUs the function will raise a ValueError.
        
        Parameters
        ----------
        new_id : int
            New GPU Id to use.
        """
        if (new_id != self._GPU_ID):
            if (new_id >= HW.NUM_GPUS):
                raise InvalidDeviceIDError(f"ID given ({new_id}) is greater than the number of available GPUs ({HW.NUM_GPUS})")
            self._GPU_ID = new_id
            self._act_funcions_manager.set_gpu_id(self._GPU_ID)
                
    def _change_COMPUTATIONAL_METHOD(self,new_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],gpu_id:int = None)->Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]:
        """
        Method to force the layer to use a specific computational method. WARNING, this will forece a kernel recompilation.
        
        Parameters
        ----------
        new_method : Literal["GPU_CUDA", "CPU_JIT", "CPU_PYTHON"]
            String of the new computational method.
        gpu_id : int, optional
            When converting from a CPU method to the GPU method can set a GPU Id of not pass it utilize the last set GPU Id.

        Returns
        -------
        Literal["GPU_CUDA", "CPU_JIT", "CPU_PYTHON"]
            The computational method that was set. This could be different from the requested one if the requested one is not available or encontered an error with out :obj:`~HeteroSymNN.Backend.hardware.WARNINGS_STRICT_MODE` been set to True.
        """
        new_method = new_method.upper()
        if not(new_method in ["GPU_CUDA","CPU_JIT","CPU_PYTHON"]):
            raise ValueError("Se intento cambiar a un metodo computacional que no es GPU_CUDA, CPU_JIT o CPU_PYTHON")
        
        if (gpu_id == None):
            gpu_id = self._GPU_ID

        if ((new_method == "GPU_CUDA") and not(HW.GPU_ENABLED)):
            if (settings.warning_level == "error"):
                raise BackendNotAvailableError("Tried to change to use 'GPU_CUDA', but no GPU is available.")
            elif (settings.warning_level == "warn"):
                warnings.warn("Tried to change to use 'GPU_CUDA', but no GPU is available. tying with CPU_JIT",PerformanceWarning,stacklevel=2)
                new_method = "CPU_JIT"
        if ((new_method == "CPU_JIT")and not(HW.CPP_JIT_ENABLED)):
            if (settings.warning_level == "error"):
                raise BackendNotAvailableError("Tried to change to use 'CPU_JIT', but no C++ compiler is available.")
            elif (settings.warning_level == "warn"):
                warnings.warn("Tried to change to use 'CPU_JIT', but no C++ compiler is available."+"Using CPU_PYTHON instead.",PerformanceWarning,stacklevel=2)
                new_method = "CPU_PYTHON"

        if(new_method != self._COMPUTATIONAL_METHOD):
            self._COMPUTATIONAL_METHOD = self._act_funcions_manager._change_method(new_method,self._GPU_ID)
            self.param_offsets = self._ASNUMPY(self.param_offsets)
            self._GPU_ID = gpu_id
            if ("CPU" in self._COMPUTATIONAL_METHOD):
                self._CALCULATION_MANAGER = np
                self._ASNUMPY = np.array
                self.param_offsets = self._CALCULATION_MANAGER.array(self.param_offsets)
            elif ("GPU" in self._COMPUTATIONAL_METHOD):
                self._CALCULATION_MANAGER = HW.cp
                self._ASNUMPY = HW.cp.array
                with HW.be.cuda.Device(self._GPU_ID):
                    self.param_offsets = self._CALCULATION_MANAGER.array(self.param_offsets)

            self.batch_size_change(self.z.shape[1])
        return self._COMPUTATIONAL_METHOD

    def to(self,device:Literal["CPU","GPU"])->None:
        """
        Change the location of the waights, biases, connection mask and activation function constants from CPU to GPU or GPU to CPU.

        Parameters
        ----------
        device : Literal["CPU", "GPU"]
            To which device to change the data of the layer.
        """
        device = device.upper()
        if not(device in ["CPU","GPU"]):
            raise ValueError("Device not recognized. Expecting CPU or GPU.")
        
        new_vector_format = self._CALCULATION_MANAGER.array
        if ((device == "GPU") and not(HW.GPU_ENABLED)):
            if (settings.warning_level == "error"):
                raise BackendNotAvailableError("Trying to send the parameters to GPU but no GPU is available.")
            elif (settings.warning_level == "warn"):
                warnings.warn("Trying to send the parameters to GPU but no GPU is available."+"Using CPU instead.",HardwareWarning,stacklevel=2)
        if (((device == "GPU") and not(HW.GPU_ENABLED)) or (device == "CPU")):
            new_vector_format = self._ASNUMPY
            device = "CPU"

        if ((device == "GPU")and("CPU" in self._COMPUTATIONAL_METHOD)):
            if (settings.warning_level == "error"):
                raise BackendNotAvailableError("Tried to send the parameters to the GPU when the CPU was set as the computational device.")
            elif (settings.warning_level == "warn"):
                warnings.warn("Tried to send the parameters to the GPU when the CPU was set as the computational device."+"Ignoring the request for safety.",HardwareWarning,stacklevel=3)
                device = "CPU"

        if(device != self._CURRENT_DEVICE):
            self._CURRENT_DEVICE = device
            self._CURRENT_VECTOR_FORMAT = new_vector_format
            if (device == "GPU"):
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

        if (self.z.shape != expected_shape):
            if ("GPU" in self._COMPUTATIONAL_METHOD):
                with HW.be.cuda.Device(self._GPU_ID):
                    self.z = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
                    self.a = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
                    self.delta = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
            else:
                self.z = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
                self.a = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
                self.delta = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)

    def forward(self,input_values:BackendArray)->BackendArray:
        """
        Method to perform the forward pass of the layer.
        
        Parameters
        ----------
        input_values : :obj:`~HeteroSymNN.types.BackendArray`
            Input values to the layer.

        Returns
        -------
        :obj:`~HeteroSymNN.types.BackendArray`
            Output values of the layer after applying the activation functions.
        """
        raise NotImplementedError

    def backward(self,error_values:BackendArray)->BackendArray:
        """
        Method to perform the backward pass of the layer.
        
        Parameters
        ----------
        error_values : :obj:`~HeteroSymNN.types.BackendArray`
            Error values from the next layer.

        Returns
        -------
        :obj:`~HeteroSymNN.types.BackendArray`
            Error values to be passed to the previous layer.
        """
        raise NotImplementedError
    
    def change_constant(self,new_values:Union[list[ConstantToUpdate],ConstantToUpdate])-> None:
        """
        Method to change the value of a activation function constant of a node or list of nodes.
        
        Parameters
        ----------
        new_values: Union[list[:obj:`~HeteroSymNN.types.ConstantToUpdate`], :obj:`~HeteroSymNN.types.ConstantToUpdate`]
            New value or list of new values to set. Each value is a tuple containing the node index, the constant name, and the new value.
        """
        if (type(new_values[0]) == int):
            new_values = [new_values]
        
        for value in new_values:
            traductor = self.CONSTANT_DICT[value[0]]
            self._funcs_constats[traductor[value[1]]] = value[2]

    def get_parameters(self)->dict[str,np.ndarray]:
        #see if the mask is geting saved
        """
        Method to get the parameters of the layer.
        
        :return: Dictionary of the parameters by string.
        :rtype: dict[str, ndarray]
        """
        raise NotImplementedError
        
    def set_parameters(self, params:dict[str,np.ndarray])->None:
        """
        Method to set the parameters of the layer.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Dictionary containing the new parameters with keys 'weights' and 'biases'.
        """
        raise NotImplementedError

class LinearLayer(BaseLayer):
    """
        Linear layer class
        
        Parameters
        ----------
        _num_inputs : int
            Number of inputs the layer is going to receive.
        layer_configuration : :obj:`~HeteroSymNN.types.LayerConstructionConfig`
            Configuration of the layer including the node activation functions and their constants as well as the initial _weights, _biases and connection mask.
        batch_size : int, optional
            Initial batch size for the layer, by default 1.
        Gpu_id : int, optional
            Id of the GPU to use if available, by default 0.
        
        Examples
        --------
        Generaly one doesn't need to instanciate this class directly but if some want to do it, here is an example of how to do it.

        >>> from HeteroSymNN.Core.Nets.layers import LinearLayer
        >>> from HeteroSymNN.Core.initializers import HeNormal
        >>> num_inputs = 3
        >>> num_nodes = 5
        >>> inicial_params = HeNormal().generate(3,5)
        >>> layer_config = [("relu", {}),("relu", {}),("sigmoid", {}),("relu", {}),("relu", {})]
        >>> constuctor = (layer_config,inicial_params)
        >>> layer = LinearLayer(num_inputs,constuctor)
        
        
    """
    def __init__(self,num_inputs:int,layer_configuration:LayerConstruction,batch_size:int = 1,Gpu_id:int = 0):
        
        init_biases, init_weights, init_mask = self._initializer.generate(num_inputs, self._num_nodes)
        self._biases = np.array(init_biases).reshape(-1,1).astype(self._DEFAULT_FLOAT_TYPE)
        self._weights = np.array(init_weights).astype(self._DEFAULT_FLOAT_TYPE)
        self._connection_mask = np.array(init_mask).astype(self._DEFAULT_FLOAT_TYPE)

        super().__init__(num_inputs,layer_configuration,batch_size,Gpu_id)
    
    #quitar la propiedad para no romper los optimizadores
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
    
    #quitar la propiedad para no romper los optimizadores
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
    
  
    def reset_parameters(self,reset_constants:bool = False,reset_mask:bool = False)->None:
        """
        Method to reset the parameters of the layer to their initial values.

        Parameters
        ----------
        reset_constants: bool, optional
            Bool value if you want to also reset the activation function constants, by default False.
        """
        super().reset_parameters(reset_constants)
        init_biases, init_weights, init_mask = self._initializer.generate(self._num_inputs, self._num_nodes)
        self._biases = np.array(init_biases).reshape(-1,1).astype(self._DEFAULT_FLOAT_TYPE)
        self._weights = np.array(init_weights).astype(self._DEFAULT_FLOAT_TYPE)
        if (reset_mask):
            self._connection_mask = np.array(init_mask).astype(self._DEFAULT_FLOAT_T)


    def to(self,device:Literal["CPU","GPU"])->None:
        """
        Change the location of the waights, biases, connection mask and activation function constants from CPU to GPU or GPU to CPU.

        Parameters
        ----------
        device : Literal["CPU", "GPU"]
            To which device to change the data of the layer.
        """
        device = device.upper()
        same = (device == self._CURRENT_DEVICE)
        super().to(device)
        if((device == self._CURRENT_DEVICE)and not (same)):
            if (device == "GPU"):
                with HW.be.cuda.Device(self._GPU_ID):
                    self._weights = self._CURRENT_VECTOR_FORMAT(self._weights)
                    self._biases = self._CURRENT_VECTOR_FORMAT(self._biases)
                    self._connection_mask = self._CURRENT_VECTOR_FORMAT(self._connection_mask)
            else:
                self._weights = self._CURRENT_VECTOR_FORMAT(self._weights)
                self._biases = self._CURRENT_VECTOR_FORMAT(self._biases)
                self._connection_mask = self._CURRENT_VECTOR_FORMAT(self._connection_mask)


    def forward(self,input_values:BackendArray)->BackendArray:
        """
        Method to perform the forward pass of the layer.
        
        Parameters
        ----------
        input_values : :obj:`~HeteroSymNN.types.BackendArray`
            Input values to the layer.

        Returns
        -------
        :obj:`~HeteroSymNN.types.BackendArray`
            Output values of the layer after applying the activation functions.
        """
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
        error_values : :obj:`~HeteroSymNN.types.BackendArray`
            Error values from the next layer.

        Returns
        -------
        :obj:`~HeteroSymNN.types.BackendArray`
            Error values to be passed to the previous layer.
        """
        batch_size = self.z.shape[1]
        self._act_funcions_manager.backward_kernel(self.z, error_values, self.delta, self._funcs_constats, self.param_offsets, self._num_nodes,batch_size)
        effective_weights = self._weights * self._connection_mask
        prev_layer_error_sum = self._CALCULATION_MANAGER.dot(effective_weights.T, self.delta)
        
        return prev_layer_error_sum

    def get_parameters(self)->dict[str,np.ndarray]:
        #see if the mask is geting saved
        """
        Method to get the parameters of the layer.
        
        :return: Dictionary of the parameters by string.
        :rtype: dict[str, ndarray]
        """
        return {
            '_weights': self._ASNUMPY(self._weights).T,
            '_biases': self._ASNUMPY(self._biases).T
        }
        
    def set_parameters(self, params:dict[str,np.ndarray])->None:
        """
        Method to set the parameters of the layer.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Dictionary containing the new parameters with keys 'weights' and 'biases'.
        """
        corret_weights = (params["_weights"].shape == self._weights.shape)
        correct_biases = (params["_biases"].shape == self._biases.shape)
        if not(correct_biases or corret_weights):
            raise LayerConfigurationError(f"""
                Weights and biases are not in the correct dimentions. Expected {self._weights.T.shape} for the weights and {self._biases.T.shape} for the biases.
                Received {params["_weights"].T.shape} for the weights and {params["_biases"].T.shape} for the biases.
                 """)
        elif not(corret_weights):
            raise LayerConfigurationError(f"""
                Weights are not in the correct dimentions. Expected {self._weights.T.shape} for the weights.
                Received {params["_weights"].T.shape} for the weights.
                """)
        elif not (correct_biases):
            raise LayerConfigurationError(f"""
                Biases are not in the correct dimentions. Expected {self._biases.T.shape} for the biases.
                Received {params["_biases"].T.shape} for the biases.
                """)
        
        self._weights = np.array(params['_weights'], dtype=self._DEFAULT_FLOAT_TYPE)
        self._biases = np.array(params['_biases'], dtype=self._DEFAULT_FLOAT_TYPE)
    
    def set_connection_mask(self, connection_mask: np.ndarray)->None:
        """
        Method to set the connection mask of the layer.

        Parameters
        ----------
        _connection_mask : np.ndarray
            2D array of ones and zeros representing the connection mask. in the of shape (num_nodes, num_inputs).
        """
        if self._CURRENT_DEVICE == 'GPU':
            self._connection_mask = self._CALCULATION_MANAGER.array(connection_mask,dtype=self._CALCULATION_MANAGER.float32)
        else:
            self._connection_mask = connection_mask.astype(self._DEFAULT_FLOAT_TYPE)

class RecurrentLayer(BaseLayer):
    """
        Recurrent layer class.
    """
    def __init__(self,num_inputs:int,layer_configuration:LayerConstruction,batch_size:int = 1,Gpu_id:int = 0):
        
        raise NotImplementedError
        super().__init__(num_inputs,layer_configuration,batch_size,Gpu_id)