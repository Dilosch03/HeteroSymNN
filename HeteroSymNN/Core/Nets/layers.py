import numpy as np
from typing import Literal, Union
import warnings

from ...Backend import hardware as HW
from ...types import LayerConstructionConfig,NodeConfig
from ...JIT.compiler import SymbolicJITCompiler

class Layer:
    def __init__(self,num_inputs:int,layer_configuration:LayerConstructionConfig,batch_size:int = 1,Gpu_id:int = 0):
        self._CALCULATION_MANAGER = HW.be
        self._ASNUMPY = HW.asnumpy
        self._GPU_ID = Gpu_id
        self.CURRENT_DEVICE = "CPU"
        self.COMPUTATIONAL_METHOD = HW.DEFAULT_COMPUTE_METHOD
        self.CURRENT_VECTOR_FORMAT = self._ASNUMPY
        self._DEFAULT_FLOAT_TYPE = self._CALCULATION_MANAGER.float32

        self.num_inputs = num_inputs
        self.num_nodes = len(layer_configuration[0])
        self.layer_node_configs = layer_configuration[0]

        init_biases, init_weights, init_mask = layer_configuration[1]
        self.biases = np.array(init_biases).reshape(-1,1).astype(self._DEFAULT_FLOAT_TYPE)
        self.weights = np.array(init_weights).astype(self._DEFAULT_FLOAT_TYPE)
        self.connection_mask = np.array(init_mask).astype(self._DEFAULT_FLOAT_TYPE)

        if (self.COMPUTATIONAL_METHOD.split("_")[0] == "GPU"):
            with HW.be.cuda.Device(self._GPU_ID):
                self._funcs_constats,self.param_offsets = self._generate_constant_array(layer_configuration[0])
        else:
            self._funcs_constats,self.param_offsets = self._generate_constant_array(layer_configuration[0])

        self._reallocate_buffers(batch_size)

        self._act_funcions_manager = SymbolicJITCompiler(layer_configuration[0],self.COMPUTATIONAL_METHOD,self._GPU_ID)

    def _generate_constant_array(self,activation_functions:list[NodeConfig]):
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
            return np.array([0.0],dtype=self._DEFAULT_FLOAT_TYPE),self._CALCULATION_MANAGER.array([0]*self.num_nodes)
        return np.array(temp,dtype=self._DEFAULT_FLOAT_TYPE),self._CALCULATION_MANAGER.array(offsets)

    def recunstruct_layer_config(self):
        self._to("CPU")
        new_layer_config = []
        for i,node in enumerate(self.layer_node_configs):
            node_constant_dict = {}
            sorted_constants = sorted(node[1].keys())
            for key in sorted_constants:
                value = self._funcs_constats[self.CONSTANT_DICT[i][key]]
                node_constant_dict[key] = float(value)

            new_layer_config.append((node[0],node_constant_dict))
        return new_layer_config
    
    def set_gpu_id(self,new_id:int):
        if (new_id != self._GPU_ID):
            if (new_id >= HW.NUM_GPUS):
                raise ValueError(f"ID de GPU {new_id} no es válido. GPUs disponibles: {HW.NUM_GPUS}")
            self._GPU_ID = new_id
            self._act_funcions_manager.set_gpu_id(self._GPU_ID)
                
    def _change_COMPUTATIONAL_METHOD(self,new_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],gpu_id:int = None):
        if not(new_method in ["GPU_CUDA","CPU_JIT","CPU_PYTHON"]):
            raise ValueError("Se intento cambiar a un metodo computacional que no es GPU_CUDA, CPU_JIT o CPU_PYTHON")
        
        if (gpu_id == None):
            gpu_id = self._GPU_ID

        if ((new_method == "GPU_CUDA") and not(HW.GPU_ENABLED)):
            if (HW.WARNINGS_STRICT_MODE):
                raise RuntimeError("Se intento cambiar al metodo de GPU_CUDA cuando no se tiene una gpu valida.")
            else:
                warnings.warn("Se intento cambiar al metodo de GPU_CUDA cuando no se tiene una gpu valida."+"Intantando con el metodo CPU_JIT")
                new_method = "CPU_JIT"
        if ((new_method == "CPU_JIT")and not(HW.CPP_JIT_ENABLED)):
            if (HW.WARNINGS_STRICT_MODE):
                raise RuntimeError("Se intento cambiar al metodo de CPU_JIT cuando no se tiene un compilador de c++ valido.")
            else:
                warnings.warn("Se intento cambiar al metodo de GPU_CUDA cuando no se tiene un compilador de c++ valido."+"Cambiando al metodo CPU_PYTHON")
                new_method = "CPU_PYTHON"

        if(new_method != self.COMPUTATIONAL_METHOD):
            self.COMPUTATIONAL_METHOD = self._act_funcions_manager._change_method(new_method,self._GPU_ID)
            self.param_offsets = self._ASNUMPY(self.param_offsets)
            self._GPU_ID = gpu_id
            if ("CPU" in self.COMPUTATIONAL_METHOD):
                self._CALCULATION_MANAGER = np
                self._ASNUMPY = np.array
                self.param_offsets = self._CALCULATION_MANAGER.array(self.param_offsets)
            elif ("GPU" in self.COMPUTATIONAL_METHOD):
                self._CALCULATION_MANAGER = HW.cp
                self._ASNUMPY = HW.cp.array
                with HW.be.cuda.Device(self._GPU_ID):
                    self.param_offsets = self._CALCULATION_MANAGER.array(self.param_offsets)

            self.batch_size_change(self.z.shape[1])
        return self.COMPUTATIONAL_METHOD

    def _to(self,device:Literal["CPU","GPU"]):
        if not(device in ["CPU","GPU"]):
            raise ValueError("Se paso como device algo que no es GPU o CPU.")
        
        new_vector_format = self._CALCULATION_MANAGER.array
        if ((device == "GPU") and not(HW.GPU_ENABLED)):
            if (HW.WARNINGS_STRICT_MODE):
                raise RuntimeError("Se intento cambiar al dispositivo GPU cuando no se tiene una gpu valida.")
            else:
                warnings.warn("Se intento cambiar al dispositivo GPU cuando no se tiene una gpu valida."+"Cambiando a CPU")
        if (((device == "GPU") and not(HW.GPU_ENABLED)) or (device == "CPU")):
            new_vector_format = self._ASNUMPY
            device = "CPU"

        if ((device == "GPU")and("CPU" in self.COMPUTATIONAL_METHOD)):
            if (HW.WARNINGS_STRICT_MODE):
                raise RuntimeError("Se intento cambiar al dispositivo GPU cuando se tiene definido la CPU como dispositivo computacional")
            else:
                warnings.warn("Se intento cambiar al dispositivo GPU cuando se tiene definido la CPU como dispositivo computacional."+"Ignorando peticion por seguridad.")
                device = "CPU"

        if(device != self.CURRENT_DEVICE):
            self.CURRENT_DEVICE = device
            self.CURRENT_VECTOR_FORMAT = new_vector_format
            if (device == "GPU"):
                with HW.be.cuda.Device(self._GPU_ID):
                    self.weights = self.CURRENT_VECTOR_FORMAT(self.weights)
                    self.biases = self.CURRENT_VECTOR_FORMAT(self.biases)
                    self.connection_mask = self.CURRENT_VECTOR_FORMAT(self.connection_mask)
                    self._funcs_constats = self.CURRENT_VECTOR_FORMAT(self._funcs_constats)
            else:
                self.weights = self.CURRENT_VECTOR_FORMAT(self.weights)
                self.biases = self.CURRENT_VECTOR_FORMAT(self.biases)
                self.connection_mask = self.CURRENT_VECTOR_FORMAT(self.connection_mask)
                self._funcs_constats = self.CURRENT_VECTOR_FORMAT(self._funcs_constats)

    def _reallocate_buffers(self, batch_size: int):
        expected_shape = (self.num_nodes, batch_size)
        if ("GPU" in self.COMPUTATIONAL_METHOD):
            with HW.be.cuda.Device(self._GPU_ID):
                self.z = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
                self.a = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
                self.delta = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
        else:
            self.z = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
            self.a = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)
            self.delta = self._CALCULATION_MANAGER.zeros(expected_shape, dtype=self._DEFAULT_FLOAT_TYPE)

    def batch_size_change(self,new_batch_size:int):
        self._reallocate_buffers(new_batch_size)

    def _ensure_batch_shape(self, batch_size: int):
        expected_shape = (self.num_nodes, batch_size)
        if self.z.shape != expected_shape:
            self._reallocate_buffers(batch_size)

    def forward(self,input_values):
        batch_size = input_values.shape[1]
        self._ensure_batch_shape(batch_size)
        effective_weights = self.weights * self.connection_mask
        self.z = self._CALCULATION_MANAGER.dot(effective_weights, input_values) + self.biases

        self._act_funcions_manager.forward_kernel(self.z, self.a, self._funcs_constats, self.param_offsets, self.num_nodes,batch_size)
        return self.a

    def backward(self,error_values):
        batch_size = self.z.shape[1]
        self._act_funcions_manager.backward_kernel(self.z, error_values, self.delta, self._funcs_constats, self.param_offsets, self.num_nodes,batch_size)
        effective_weights = self.weights * self.connection_mask
        prev_layer_error_sum = self._CALCULATION_MANAGER.dot(effective_weights.T, self.delta)
        
        return prev_layer_error_sum
    
    def change_constant(self,new_values:Union[list[tuple[int,str,float]],tuple[int,str,float]]):
        if (type(new_values[0]) == int):
            new_values = [new_values]
        
        for value in new_values:
            traductor = self.CONSTANT_DICT[value[0]]
            self._funcs_constats[traductor[value[1]]] = value[2]

    def get_parameters(self):
        return {
            'weights': self._ASNUMPY(self.weights).T,
            'biases': self._ASNUMPY(self.biases).T
        }

    def set_parameters(self, params:dict[str,np.ndarray]):
        corret_weights = (params["weights"].shape == self.weights.shape)
        correct_biases = (params["biases"].shape == self.biases.shape)
        if not(correct_biases or corret_weights):
            raise ValueError(f"""Pesos y biases nuevos no estan en las dimenciones correctas.Pesos esperaba {self.weights.T.shape} y 
                             recibió {params["weights"].T.shape}. Biases esperaba {self.biases.T.shape} y recibió {params["biases"].T.shape}.""")
        elif not(corret_weights):
            raise ValueError(f"""Pesos nuevos no estan en las dimenciones correctas. 
                             Pesos esperaba {self.weights.T.shape} y 
                             recibió {params["weights"].T.shape}""")
        elif not (correct_biases):
            raise ValueError(f"""Biases nuevos no esta en la dimencion correcta.Biases esperaba 
                             {self.biases.T.shape} y recibió {params["biases"].T.shape}.""")
        
        self.weights = np.array(params['weights'], dtype=self._DEFAULT_FLOAT_TYPE)
        self.biases = np.array(params['biases'], dtype=self._DEFAULT_FLOAT_TYPE)
    
    def set_connection_mask(self, connection_mask: np.ndarray):
        if self.CURRENT_DEVICE == 'GPU':
            self.connection_mask = self._CALCULATION_MANAGER.array(connection_mask,dtype=self._CALCULATION_MANAGER.float32)
        else:
            self.connection_mask = connection_mask.astype(self._DEFAULT_FLOAT_TYPE)