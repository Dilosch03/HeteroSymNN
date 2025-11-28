from typing import Optional,Literal
import warnings
import numpy as np

from ...Backend import hardware as HW
from ...types import NodeConfig,LayerValues,LayerConstructionConfig,FlexibleNodeConfig
from .layers import Layer
from .. import losses as lossC, optimizers as OptiC, initializers as InitC

class ConfigurableNN:
    def __init__(self, nodes_structure: list[int], detailed_activations: list[list[NodeConfig]],initial_values: Optional[list[LayerValues]] = None,initializer: Optional[InitC.Initializer] = None,
                 learning_rate: float = 0.001, batch_size: int = 32, training_mode: Literal["batch", "mini-batch", "stochastic"] = "mini-batch", learning_mode: str = "Static",
                 loss_function: lossC = None, optimizer: Optional[OptiC.Optimizer] = None, num_treaning_iter: int = 1000):
        
        self._CALCULATION_MANAGER = HW.be
        self._ASNUMPY = HW.asnumpy
        self.num_complited_train_iterations = 0

        if len(nodes_structure) < 2:
            raise ValueError("nodes_structure debe tener al menos 2 elementos (entrada y salida).")
        
        num_layers = len(nodes_structure) - 1
        if (len(detailed_activations) != num_layers):
            raise ValueError(f"La estructura de nodos define {num_layers} capas (conexiones), pero 'detailed_activations' tiene {len(detailed_activations)} elementos.")
        
        if ((initial_values is not None) and (len(initial_values) != num_layers)):
            raise ValueError(f"Se proporcionaron valores iniciales, pero su longitud ({len(initial_values)}) no coincide con el número de capas ({num_layers}).")

        self._LEARNING_RATE = learning_rate
        self._LEAR_MODE = learning_mode
        self._TRAIN_MODE =training_mode
        self._BATCH_SIZE = batch_size
        self._GPU_ID = 0
        self._DEFAULT_FLOAT_TYPE = self._CALCULATION_MANAGER.float32
        self.CURRENT_DEVICE = "CPU"
        self.COMPUTATIONAL_METHOD = HW.DEFAULT_COMPUTE_METHOD
        self.num_treaning_iterations = num_treaning_iter
        self.NODE_STRUCTURE = nodes_structure
        self.NODE_CONFIGS = detailed_activations

        if (initializer is None):
            self.INITIALIZER = InitC.HeNormal()
        else:
            self.INITIALIZER = initializer

        if (loss_function == None):
            self.LOSS_FUNCTION:lossC.Loss = lossC.MSELoss(self.COMPUTATIONAL_METHOD,self._GPU_ID)
        else:
            self.LOSS_FUNCTION = loss_function
            temp_result = self.LOSS_FUNCTION._change_COMPUTATIONAL_METHOD(self.COMPUTATIONAL_METHOD,self._GPU_ID)
            if (temp_result != self.COMPUTATIONAL_METHOD):
                raise RuntimeError(f"Funcion de perdia no pudo sincronizar su metodo al de la red.Metodo de la red:{self.COMPUTATIONAL_METHOD}. Metodo de la funcion de perdida: {temp_result}")

        if (optimizer != None):
            self.UPDATE_METHOD = optimizer
            if (self.UPDATE_METHOD.learning_rate == None):
                self.UPDATE_METHOD.learning_rate = self._LEARNING_RATE
            else:
                self._LEARNING_RATE = self.UPDATE_METHOD.learning_rate
        else:
            self.UPDATE_METHOD:OptiC.Optimizer = OptiC.AdamOptimizer(self._LEARNING_RATE,self.COMPUTATIONAL_METHOD.split("_")[0],device_id=self._GPU_ID)

        if self._TRAIN_MODE == "stochastic":
            self._BATCH_SIZE = 1
        
        self.layers: list[Layer] = []

        for i in range(num_layers):
            num_inputs = nodes_structure[i]
            num_nodes = nodes_structure[i+1]
            
            node_configs = detailed_activations[i]
            
            if (len(node_configs) != num_nodes):
                raise ValueError(f"Error en Capa {i+1}: Se definieron {num_nodes} nodos en 'nodes_structure', pero hay {len(node_configs)} configuraciones de activación.")

            # Determinar valores iniciales (Pasados o Generados)
            if (initial_values is not None):
                current_vals = initial_values[i]
            else:
                current_vals = self.INITIALIZER.generate(num_inputs, num_nodes)

            # Crear la capa
            layer_config: LayerConstructionConfig = (node_configs, current_vals)
            self.layers.append(Layer(num_inputs, layer_config, self._BATCH_SIZE, self._GPU_ID))


        self.histogram_losses = []    
    
    @property
    def learning_rate(self):
        return self._LEARNING_RATE
    
    @learning_rate.setter
    def learning_rate(self,new_learning_rate:float):
        self._LEARNING_RATE = new_learning_rate
        self.UPDATE_METHOD.learning_rate = new_learning_rate

    def set_gpu_id(self,new_id:int):
        if (self._GPU_ID != new_id):
            if (new_id >= HW.NUM_GPUS):
                raise ValueError(f"ID de GPU {new_id} no es válido. GPUs disponibles: {HW.NUM_GPUS}")
            self.change_device("CPU")
            self._GPU_ID = new_id
            self.LOSS_FUNCTION.set_gpu_id(self._GPU_ID)
            self.UPDATE_METHOD.set_gpu_id(self._GPU_ID)
            for layer in self.layers:
                layer.set_gpu_id(new_id)

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
                self._to("CPU")
                self.COMPUTATIONAL_METHOD = new_method
                self._GPU_ID = gpu_id
                if ("CPU" in new_method):
                    self._CALCULATION_MANAGER = np
                    self._ASNUMPY = np.array
                elif ("GPU" in new_method):
                    self._CALCULATION_MANAGER = HW.cp
                    self._ASNUMPY = HW.cp.array
                
                temp_result = self.LOSS_FUNCTION._change_COMPUTATIONAL_METHOD(self.COMPUTATIONAL_METHOD,self._GPU_ID)
                if (temp_result != self.COMPUTATIONAL_METHOD):
                    raise RuntimeError(f"Funcion de perdia no pudo sincronizar su metodo al de la red.Metodo de la red:{self.COMPUTATIONAL_METHOD}. Metodo de la funcion de perdida: {temp_result}")
                self.UPDATE_METHOD._change_COMPUTACIONAL_DEVICE(self.COMPUTATIONAL_METHOD.split("_")[0])
                laye_calc_method = []
                for layer in self.layers:
                    laye_calc_method.append(layer._change_COMPUTATIONAL_METHOD(new_method,gpu_id))
                
                expected = [new_method]*len(self.layers)
                if (laye_calc_method != expected):
                    raise RuntimeError(f"Cambio de metodo no se pudo realizar. Uno o mas capas no estan en el nuevo metodo. Configuracion nueva de las capas: {laye_calc_method}")


    def change_device(self, device:Literal["CPU","GPU"]):
        if not(device in ["CPU","GPU"]):
            raise ValueError("Se paso como device algo que no es GPU o CPU.")
        
        if ((device == "GPU") and not(HW.GPU_ENABLED)):
            if (HW.WARNINGS_STRICT_MODE):
                raise RuntimeError("Se intento cambiar al dispositivo GPU cuando no se tiene una gpu valida.")
            else:
                warnings.warn("Se intento cambiar al dispositivo GPU cuando no se tiene una gpu valida."+"Cambiando a CPU")
                device = "CPU"

        if ((device == "GPU")and("CPU" in self.COMPUTATIONAL_METHOD)):
            if (HW.WARNINGS_STRICT_MODE):
                raise RuntimeError("Se intento cambiar al dispositivo GPU cuando se tiene definido la CPU como dispositivo computacional")
            else:
                warnings.warn("Se intento cambiar al dispositivo GPU cuando se tiene definido la CPU como dispositivo computacional."+"Ignorando peticion por seguridad.")
                device = "CPU"
        if(device != self.CURRENT_DEVICE):
                self.CURRENT_DEVICE = device
                self._to(device)


    def _to(self, device:Literal["CPU","GPU"]):
        self.UPDATE_METHOD._to_device(device)
        for layer in self.layers:
            layer._to(device)

    def forward(self,input_values:list,to_cpu:bool = True):
        self.change_device(self.COMPUTATIONAL_METHOD.split("_")[0])
        if not isinstance(input_values, (np.ndarray, self._CALCULATION_MANAGER.ndarray)):
            current_a = self._CALCULATION_MANAGER.array(input_values, dtype=self._DEFAULT_FLOAT_TYPE)
        else:
            current_a = self._CALCULATION_MANAGER.asarray(input_values, dtype=self._DEFAULT_FLOAT_TYPE)
            
        if current_a.ndim == 1:
            current_a = current_a.reshape(-1, 1)
        

        if current_a.shape[0] == self.layers[0].num_inputs:
            pass
        elif current_a.shape[1] == self.layers[0].num_inputs:
            current_a = current_a.T
        else:
            raise ValueError(
                f"La forma de los datos de entrada {current_a.shape} es incorrecta. "
                f"La Capa 0 esperaba {self.layers[0].num_inputs} características (features), "
                f"pero ninguna dimensión ({current_a.shape[0]} o {current_a.shape[1]}) coincidió."
            )
            

        for layer in self.layers:
            current_a = layer.forward(current_a)

        if(to_cpu):
            return self._ASNUMPY(current_a).T

        return current_a

    def backward(self,error_values):
        self.change_device(self.COMPUTATIONAL_METHOD.split("_")[0])
        next_layer_error_sum =self._CALCULATION_MANAGER.array(error_values, dtype=self._CALCULATION_MANAGER.float32)
        
        for layer in reversed(self.layers):
            next_layer_error_sum = layer.backward(next_layer_error_sum)

    def train_step(self, x_input: list, y_target: list):
        self.change_device(self.COMPUTATIONAL_METHOD.split("_")[0])
        self.num_complited_train_iterations += 1
        
        y_pred = self.forward(x_input,False)
        
        loss = self.LOSS_FUNCTION.forward(y_pred, y_target)
        error_to_propagate = self.LOSS_FUNCTION.backward(y_pred, y_target)

        self.backward(error_to_propagate)
        self.update_params(x_input)

        return loss
    
    def update_params(self, inputs):
        self.change_device(self.COMPUTATIONAL_METHOD.split("_")[0])
        self.UPDATE_METHOD.step(self.layers,inputs) 

    def train(self,training_inputs: list[list[float]], training_targets: list[list[float]],num_iterations = None,
              training_mode: Literal["batch", "mini-batch", "stochastic"] = None, batch_size: int = None):
        self.change_device(self.COMPUTATIONAL_METHOD.split("_")[0])
        train_data = self._CALCULATION_MANAGER.array(training_inputs,dtype=np.float32).T
        train_targets = self._CALCULATION_MANAGER.array(training_targets,dtype=np.float32).T

        mode = self._TRAIN_MODE if training_mode is None else training_mode
        b_size = self._BATCH_SIZE if batch_size is None else batch_size
        if (num_iterations == None):
            num_iterations = self.num_treaning_iterations

        self._TRAIN_MODE = mode
        if (b_size != self._BATCH_SIZE):
            self._BATCH_SIZE = b_size
            for layer in self.layers:
                layer.batch_size_change(b_size)

        if mode == "stochastic":
            b_size = 1
        elif mode == "batch":
            b_size = len(training_inputs)
        
        num_samples = train_data.shape[1]
        for _ in range(num_iterations):
            iter_loss = self._CALCULATION_MANAGER.array(0.0, dtype=self._DEFAULT_FLOAT_TYPE)
            indices = self._CALCULATION_MANAGER.random.permutation(num_samples)

            for start_idx in range(0, num_samples, b_size):
                end_idx = min(start_idx + b_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                x = train_data[:, batch_indices]
                y = train_targets[:, batch_indices]
                
                loss = self.train_step(x, y)
                
                iter_loss += loss * (end_idx - start_idx)

            avg_loss = self._ASNUMPY(iter_loss) / num_samples
            self.histogram_losses.append(avg_loss)

    def get_parameters(self):
        self.change_device("CPU")
        return {f'layer_{i}': layer.get_parameters() for i, layer in enumerate(self.layers)}

    def set_parameters(self, params):
        self.change_device("CPU")
        for i, layer in enumerate(self.layers):
            layer_params = params[f'layer_{i}']
            if layer_params.ndim == 0:
                layer.set_parameters(layer_params.item())
            else:
                layer.set_parameters(layer_params)

    def get_config(self):
        self.change_device("CPU")
        config = {
            'nodes_structure': self.NODE_STRUCTURE,
            'detailed_activations': self.NODE_CONFIGS,
            'learning_rate': self._LEARNING_RATE,
            'learning_mode': self._LEAR_MODE,
            'training_mode': self._TRAIN_MODE,
            'batch_size': self._BATCH_SIZE
        }

        config.update({'initializer_config': self.INITIALIZER.get_config()})
        config['optimizer_config'] = self.UPDATE_METHOD.get_config()
        config["loss_config"] = self.LOSS_FUNCTION.get_config()
        config['num_treaning_iterations'] = self.num_treaning_iterations
        return config


class FlexibleNN(ConfigurableNN):
    def __init__(self, nodes_structure: list[int], activation_config: list[FlexibleNodeConfig],initial_values: Optional[list[LayerValues]] = None,initializer: Optional[InitC.Initializer] = None,
                 learning_rate: float = 0.001, learning_mode: str = "Static", training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic", batch_size: int = 1,
                 loss_function: lossC = None, optimizer: OptiC.Optimizer = None, num_treaning_iter: int = 1000):
        
        num_layers = len(nodes_structure) - 1
        
        if not isinstance(activation_config, list):
             raise ValueError(f"activation_config debe ser una lista con un elemento por capa. Se recibió: {type(activation_config)}")
        
        if len(activation_config) != num_layers:
             raise ValueError(f"La lista de activaciones tiene {len(activation_config)} elementos, pero hay {num_layers} capas en nodes_structure.")

        detailed_activations = self._expand_to_detailed(num_layers, nodes_structure[1:], activation_config)

        super().__init__(
            nodes_structure=nodes_structure,
            detailed_activations=detailed_activations,
            initial_values=initial_values,
            initializer=initializer,
            learning_rate=learning_rate,
            learning_mode=learning_mode,
            training_mode=training_mode,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer=optimizer, 
            num_treaning_iter=num_treaning_iter
        )

    def _process_node_config(self, config_item: FlexibleNodeConfig) -> NodeConfig:
        if (isinstance(config_item, str)):
            return (config_item, {})
        elif ((isinstance(config_item, tuple)) and (len(config_item) == 2)):
            return config_item
        else:
            raise ValueError(
                f"Formato inválido para la activación: {config_item}.\n"+
                f"Se esperaba 'str' o 'tuple[str, dict[str, float]]'.\n"+
                "Ejemplos: 'relu', ('mish', {'beta': 1.0})"
            )
        
    def _expand_to_detailed(self, num_layers: int, nodes_per_layer: list[int], layer_configs: list[FlexibleNodeConfig]) -> list[list[NodeConfig]]:
        final_config = []
        for i in range(num_layers):
            layer_conf_raw = layer_configs[i]
            num_nodes = nodes_per_layer[i]
            
            # Normalizamos a tupla estricta
            node_conf = self._process_node_config(layer_conf_raw)
            
            # Repetimos para todos los nodos de la capa
            final_config.append([node_conf] * num_nodes)
            
        return final_config
    

class SimpleNN(FlexibleNN):
    def __init__(self, nodes_structure: list[int], activation: FlexibleNodeConfig = "relu", output_activation: FlexibleNodeConfig = "num",initializer: Optional[InitC.Initializer] = None,
                 learning_rate: float = 0.001, learning_mode: str = "Static",training_mode: Literal["batch", "mini-batch", "stochastic"] = "stochastic",
                 batch_size: int = 1, loss_function: lossC = None, optimizer: OptiC.Optimizer = None, num_treaning_iter: int = 1000):

        if len(nodes_structure) < 2:
            raise ValueError("nodes_structure debe tener al menos 2 elementos (entrada y salida)")

        num_hidden_layers = len(nodes_structure) - 2 
        
        activations_list = [activation] * num_hidden_layers
        activations_list.append(output_activation)

        super().__init__(
            nodes_structure=nodes_structure,
            activation_config=activations_list,
            initializer=initializer, 
            learning_rate=learning_rate,
            learning_mode=learning_mode,
            training_mode=training_mode,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer=optimizer,
            num_treaning_iter=num_treaning_iter
        )