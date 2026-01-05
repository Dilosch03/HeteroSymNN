from __future__ import annotations
from typing import Optional,Literal,Union,Any
import warnings
import numpy as np

from ...Backend import hardware as HW
from ...types import NodeConfig,LayerValues,LayerConstructionConfig,FlexibleNodeConfig,BackendArray,ConstantToUpdate
from .layers import Layer
from .. import losses as lossC, optimizers as OptiC, initializers as InitC

class ConfigurableNN:
    """
        Base class for creating a dense neural network with customizable architecture, activation functions, and training parameters.
        
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
        learning_mode : str, optional
            Learning mode of the network. Currently only "Static" is supported., by default "Static"
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
        
        self._CALCULATION_MANAGER = HW.be
        self._ASNUMPY = HW.asnumpy
        self.num_complited_train_iterations = 0
        self.num_completed_epochs = 0

        if len(nodes_structure) < 2:
            raise ValueError("nodes_structure debe tener al menos 2 elementos (entrada y salida).")
        
        num_layers = len(nodes_structure) - 1
        if (len(detailed_activations) != num_layers):
            raise ValueError(f"La estructura de nodos define {num_layers} capas (conexiones), pero 'detailed_activations' tiene {len(detailed_activations)} elementos.")
        
        if ((initial_values is not None) and (len(initial_values) != num_layers)):
            raise ValueError(f"Se proporcionaron valores iniciales, pero su longitud ({len(initial_values)}) no coincide con el número de capas ({num_layers}).")

        self._LEARNING_RATE = learning_rate
        self._LEAR_MODE = learning_mode
        self.training_mode =training_mode
        self._BATCH_SIZE = batch_size
        self._GPU_ID = 0
        self._DEFAULT_FLOAT_TYPE = self._CALCULATION_MANAGER.float32
        self._CURRENT_DEVICE = "CPU"
        self._COMPUTATIONAL_METHOD = HW.DEFAULT_COMPUTE_METHOD
        self.num_treaning_iterations = num_treaning_iter
        self._NODE_STRUCTURE = nodes_structure

        if (initializer is None):
            self._INITIALIZER = InitC.HeNormal()
        else:
            self._INITIALIZER = initializer

        if (loss_function == None):
            self._LOSS_FUNCTION:lossC.Loss = lossC.MSELoss(self._COMPUTATIONAL_METHOD,self._GPU_ID)
        else:
            self._LOSS_FUNCTION = loss_function
            temp_result = self._LOSS_FUNCTION._change_COMPUTATIONAL_METHOD(self._COMPUTATIONAL_METHOD,self._GPU_ID)
            if (temp_result != self._COMPUTATIONAL_METHOD):
                raise RuntimeError(f"Funcion de perdia no pudo sincronizar su metodo al de la red.Metodo de la red:{self._COMPUTATIONAL_METHOD}. Metodo de la funcion de perdida: {temp_result}")

        if (optimizer != None):
            self._UPDATE_METHOD = optimizer
            if (self._UPDATE_METHOD.learning_rate != None):
                self._UPDATE_METHOD.learning_rate = self._LEARNING_RATE
            else:
                self._LEARNING_RATE = self._UPDATE_METHOD.learning_rate
        else:
            self._UPDATE_METHOD:OptiC.Optimizer = OptiC.AdamOptimizer(self._LEARNING_RATE,self._COMPUTATIONAL_METHOD.split("_")[0],device_id=self._GPU_ID)

        if self.training_mode == "stochastic":
            self._BATCH_SIZE = 1
        
        self._LAYERS: list[Layer] = []

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
                current_vals = self._INITIALIZER.generate(num_inputs, num_nodes)

            # Crear la capa
            layer_config: LayerConstructionConfig = (node_configs, current_vals)
            self._LAYERS.append(Layer(num_inputs, layer_config, self._BATCH_SIZE, self._GPU_ID))


        self.histogram_losses = []    
    
    @property
    def LAYERS(self)->list[Layer]:
        """
        Property to get the layers of the network. Read-only.
        
        Returns
        -------
        list[:obj:`~HeteroSymNN.Core.Nets.layers.Layer`]
        """
        return self._LAYERS
    
    @property
    def NODE_STRUCTURE(self)->list[int]:
        """
        Property to get the number of nodes per layer of the network. Read-only.
        
        Returns
        -------
        list[int]
        """
        return self._NODE_STRUCTURE
    
    @property
    def GPU_ID(self)->int:
        """
        Property to get the current GPU ID being used by the network. Read-only.
        
        For setting a new GPU ID, use the :obj:`~set_gpu_id` method.

        Returns
        -------
        int
        """
        return self._GPU_ID
    
    @property
    def BATCH_SIZE(self)->int:
        """
        Property to get the current batch size being used by the network in case of mini-batch training. Read-only.
        
        Returns
        -------
        int
        """
        return self._BATCH_SIZE
    
    @property
    def CURRENT_DEVICE(self)->Literal["CPU","GPU"]:
        """
        Property to get the current device where the network parameters are located. Read-only.

        For changing location of the network parameters use the :obj:`~change_device` method.
        
        Returns
        -------
        Literal["CPU","GPU"]
        """
        return self._CURRENT_DEVICE
    
    @property
    def COMPUTATIONAL_METHOD(self)->Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]:
        """
        Property to get the current computational method being used by the network. Read-only.

        To change the computational method use the :obj:`~_change_COMPUTATIONAL_METHOD` method.
        
        Returns
        -------
        Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]
        """
        return self._COMPUTATIONAL_METHOD

    @property
    def OPTIMIZER(self)->OptiC.Optimizer:
        """
        Property to get the current optimizer being used by the network. Read-only.

        Returns
        -------
        :obj:`~HeteroSymNN.Core.Nets.optimizers.Optimizer`
        """
        return self._UPDATE_METHOD

    @property
    def LOSS_FUNCTION(self)->lossC.Loss:
        """
        Property to get the current loss function being used by the network. Read-only.

        Returns
        -------
        :obj:`~HeteroSymNN.Core.Nets.losses.Loss`
        """
        return self._LOSS_FUNCTION
    
    @property
    def INICIALIZER(self)->InitC.Initializer:
        """
        Property to get the current initializer being used by the network. Read-only.

        Returns
        -------
        :obj:`~HeteroSymNN.Core.Nets.initializers.Initializer`
        """
        return self._INITIALIZER

    @property
    def learning_rate(self)->float:
        """
        Property to get the current learning rate of the network. Read-write.
        
        Returns
        -------
        float
        """
        return self._LEARNING_RATE
    
    @learning_rate.setter
    def learning_rate(self,new_learning_rate:float):
        self._LEARNING_RATE = new_learning_rate
        self._UPDATE_METHOD.learning_rate = new_learning_rate

    def set_gpu_id(self,new_id:int)->None:
        """
        Method to set a new GPU ID for the network. This will change the device of the network parameters if currently on GPU.

        Parameters
        ----------
        new_id : int
            New GPU ID to set.
        
        Raises
        ------
        ValueError
            If the Id for the new GPU is greater than the number of available GPUs.
        """
        if (self._GPU_ID != new_id):
            if (new_id >= HW.NUM_GPUS):
                raise ValueError(f"ID de GPU {new_id} no es válido. GPUs disponibles: {HW.NUM_GPUS}")
            self.change_device("CPU")
            self._GPU_ID = new_id
            self._LOSS_FUNCTION.set_gpu_id(self._GPU_ID)
            self._UPDATE_METHOD.set_gpu_id(self._GPU_ID)
            for layer in self._LAYERS:
                layer.set_gpu_id(new_id)

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
        ValueError
            If the new method is not one of "GPU_CUDA", "CPU_JIT", or "CPU_PYTHON".
        RuntimeError
            If trying to set "GPU_CUDA" without a valid GPU or "CPU_JIT" without a valid C++ compiler when strict warnings mode is enabled. If not enabled, it will fallback to the next available method and throw a warning.
        RuntimeError
            In the case the loss function or any of the layers could not change to the new computational method will show a runtime error.
        
        .. Warning::
            When changing computational methods it will force a kernel recompilation for all layers and reallocation of all parameters. Depending on the size of the network this could take a while.
        """
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

        if(new_method != self._COMPUTATIONAL_METHOD):
                self._to("CPU")
                self._COMPUTATIONAL_METHOD = new_method
                self._GPU_ID = gpu_id
                if ("CPU" in new_method):
                    self._CALCULATION_MANAGER = np
                    self._ASNUMPY = np.array
                elif ("GPU" in new_method):
                    self._CALCULATION_MANAGER = HW.cp
                    self._ASNUMPY = HW.cp.array
                
                temp_result = self._LOSS_FUNCTION._change_COMPUTATIONAL_METHOD(self._COMPUTATIONAL_METHOD,self._GPU_ID)
                if (temp_result != self._COMPUTATIONAL_METHOD):
                    raise RuntimeError(f"Funcion de perdia no pudo sincronizar su metodo al de la red.Metodo de la red:{self._COMPUTATIONAL_METHOD}. Metodo de la funcion de perdida: {temp_result}")
                self._UPDATE_METHOD._change_COMPUTATIONAL_DEVICE(self._COMPUTATIONAL_METHOD.split("_")[0])
                laye_calc_method = []
                for layer in self._LAYERS:
                    laye_calc_method.append(layer._change_COMPUTATIONAL_METHOD(new_method,gpu_id))
                
                expected = [new_method]*len(self._LAYERS)
                if (laye_calc_method != expected):
                    raise RuntimeError(f"Cambio de metodo no se pudo realizar. Uno o mas capas no estan en el nuevo metodo. Configuracion nueva de las capas: {laye_calc_method}")


    def change_device(self, device:Literal["CPU","GPU"])->None:
        """
        Change location of the network parameters to the specified device.

        Parameters
        ----------
        device : Literal["CPU","GPU"]
            Device to move the network parameters to.
        """
        if not(device in ["CPU","GPU"]):
            raise ValueError("Se paso como device algo que no es GPU o CPU.")
        
        if ((device == "GPU") and not(HW.GPU_ENABLED)):
            if (HW.WARNINGS_STRICT_MODE):
                raise RuntimeError("Se intento cambiar al dispositivo GPU cuando no se tiene una gpu valida.")
            else:
                warnings.warn("Se intento cambiar al dispositivo GPU cuando no se tiene una gpu valida."+"Cambiando a CPU")
                device = "CPU"

        if ((device == "GPU")and("CPU" in self._COMPUTATIONAL_METHOD)):
            if (HW.WARNINGS_STRICT_MODE):
                raise RuntimeError("Se intento cambiar al dispositivo GPU cuando se tiene definido la CPU como dispositivo computacional")
            else:
                warnings.warn("Se intento cambiar al dispositivo GPU cuando se tiene definido la CPU como dispositivo computacional."+"Ignorando peticion por seguridad.")
                device = "CPU"
        if(device != self._CURRENT_DEVICE):
                self._CURRENT_DEVICE = device
                self._to(device)


    def _to(self, device:Literal["CPU","GPU"]):
        """
        Internal method to move the network parameters to the specified device.
        """
        self._UPDATE_METHOD._to_device(device)
        for layer in self._LAYERS:
            layer._to(device)

    def _forward(self,input_values:BackendArray)->BackendArray:
        """
        Internal method to perform a forward pass through the network.
        
        Parameters
        ----------
        input_values : :obj:`~HeteroSymNN.types.BackendArray`
            Input values for the network.
        
        Returns
        -------
        :obj:`~HeteroSymNN.types.BackendArray`
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
        error_values : :obj:`~HeteroSymNN.types.BackendArray`
            Error values to propagate back through the network.

        Returns
        -------
        :obj:`~HeteroSymNN.types.BackendArray`
            Error values propagated back to the input layer.
        """
        self.change_device(self._COMPUTATIONAL_METHOD.split("_")[0])
        next_layer_error_sum =self._CALCULATION_MANAGER.array(error_values, dtype=self._CALCULATION_MANAGER.float32)
        
        for layer in reversed(self._LAYERS):
            next_layer_error_sum = layer.backward(next_layer_error_sum)

    def train_step(self, x_input: BackendArray, y_target: BackendArray)->float:
        """
        Perform a single training step (forward pass, loss computation, backward pass, and parameter update).

        Parameters
        ----------
        x_input : :obj:`~HeteroSymNN.types.BackendArray`
            Input values to train the network on.
        y_target : :obj:`~HeteroSymNN.types.BackendArray`
            Target output values for the network.
        
        Returns
        -------
        float
            Computed loss for the training step.
        """
        self.change_device(self._COMPUTATIONAL_METHOD.split("_")[0])
        self.num_complited_train_iterations += 1
        
        y_pred = self._forward(x_input)
        
        loss = self._LOSS_FUNCTION.forward(y_pred, y_target)
        error_to_propagate = self._LOSS_FUNCTION.backward(y_pred, y_target)

        self.backward(error_to_propagate)
        self.update_params(x_input)

        return loss
    
    def update_params(self, inputs:BackendArray)->None:
        """
        Update the network parameters using the optimizer.

        Parameters
        ----------
        inputs : :obj:`~HeteroSymNN.types.BackendArray`
            Input values used for the parameter update.
        """
        self.change_device(self._COMPUTATIONAL_METHOD.split("_")[0])
        self._UPDATE_METHOD.step(self._LAYERS,inputs) 

    def train(self,training_inputs: list[list[float]], training_targets: list[list[float]],num_iterations = None,
              training_mode: Literal["batch", "mini-batch", "stochastic"] = None, batch_size: int = None)->list[float]:
        """
        Train the neural network using the provided training data.

        Parameters
        ----------
        training_inputs : list[list[float]]
            List of input samples for training. Shape should be (num_samples, num_features).
        training_targets : list[list[float]]
            List of target output samples for training. Shape should be (num_samples, num_outputs).
        num_iterations : int, optional
            Number of training iterations (epochs) to perform. If not provided, uses the value of the attribute :obj:`~num_treaning_iterations`., by default None
        training_mode : Literal["batch", "mini-batch", "stochastic"], optional
            Training mode to use during training. Options are "batch", "mini-batch", and "stochastic". If not provided, uses the current value of the attribute :obj:`~training_mode`., by default None
        batch_size : int, optional
            Batch size to use during training if training_mode is "mini-batch". If not provided, uses the current value of the attribute :obj:`~batch_size`., by default None
        
        Returns
        -------
        list[float]
            List of loss values recorded at each epoch during training.
        """
        self.change_device(self._COMPUTATIONAL_METHOD.split("_")[0])
        train_data = self._CALCULATION_MANAGER.array(training_inputs,dtype=np.float32).T
        train_targets = self._CALCULATION_MANAGER.array(training_targets,dtype=np.float32).T

        mode = self.training_mode if training_mode is None else training_mode
        b_size = self._BATCH_SIZE if batch_size is None else batch_size
        if (num_iterations == None):
            num_iterations = self.num_treaning_iterations

        self.training_mode = mode
        if (b_size != self._BATCH_SIZE):
            self._BATCH_SIZE = b_size
            for layer in self._LAYERS:
                layer.batch_size_change(b_size)

        if mode == "stochastic":
            b_size = 1
        elif mode == "batch":
            b_size = len(training_inputs)
        
        num_samples = train_data.shape[1]
        for _ in range(num_iterations):
            self.num_completed_epochs += 1
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

        return self.histogram_losses
    
    def predict(self,input_values:Union[list,list[list]],to_cpu:bool = True)->Union[np.ndarray,BackendArray]:
        """
        Make predictions using the neural network.
        
        Parameters
        ----------
        input_values : list or list[list]
            Input values for making predictions. In case of multiple samples, shape should be (num_samples, num_features) or (num_features, num_samples).
        to_cpu : bool, optional
            Whether to return the predictions as a NumPy array on the CPU. If False, returns in the current backend array format., by default True
        
        Returns
        -------
        np.ndarray or :obj:`~HeteroSymNN.types.BackendArray`
            Predicted output values.
        """
        self.change_device(self._COMPUTATIONAL_METHOD.split("_")[0])
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
            raise ValueError(
                f"La forma de los datos de entrada {current_a.shape} es incorrecta. "
                f"La Capa 0 esperaba {self._LAYERS[0].num_inputs} características (features), "
                f"pero ninguna dimensión ({current_a.shape[0]} o {current_a.shape[1]}) coincidió."
            )
        
        prediction = self._forward(current_a)
        if(to_cpu):
            return self._ASNUMPY(prediction).T
        
        return prediction

    def get_parameters(self)->dict[Union[str,int],dict[str,np.ndarray]]:
        """
        Get the parameters (weights and biases) of the network.
        
        Returns
        -------
        dict[Union[str,int],dict[str,np.ndarray]]
            Dictionary containing the parameters of each layer.
        """
        self.change_device("CPU")
        return {f'layer_{i}': layer.get_parameters() for i, layer in enumerate(self._LAYERS)}

    def set_parameters(self, params:dict[Union[str,int],dict[str,np.ndarray]])->None:
        """
        Set the parameters (weights and biases) of the network.
        
        Parameters
        ----------
        params : dict[Union[str,int],dict[str,np.ndarray]]
            Dictionary containing the parameters for each layer.
        """
        self.change_device("CPU")
        for key in params:
            layer_params = {}
            layer_params.update({"weights":params[key]["weights"].copy().T})
            layer_params.update({"biases":params[key]["biases"].copy().reshape(-1,1)})
            if (type(key) != int):
                index = int(key.split("_")[-1])
            self._LAYERS[index].set_parameters(layer_params)

    def change_constants(self,new_constants:dict[int,Union[list[ConstantToUpdate],ConstantToUpdate]])->None:
        """
        Change the constants in the activation functions of the network layers.
        
        Parameters
        ----------
        new_constants : dict[int,Union[list[:obj:`~HeteroSymNN.types.ConstantToUpdate`], :obj:`~HeteroSymNN.types.ConstantToUpdate`]]
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
                * **"nodes_structure"** (*list[int]*): List of number of nodes per layer.
                * **"detailed_activations"** (*list[list[NodeConfig]*): Activation configuration for each node.
                * **"learning_rate"** (*float*): Learning rate of the network.
                * **"learning_mode"** (*str*): Learning mode of the network.
                * **"training_mode"** (*str*): Training mode ("batch", "mini-batch", "stochastic").
                * **"batch_size"** (*int*): Batch size used during training.
                * **"initializer_config"** (*dict[str, Any]*): Configuration of the initializer.
                * **"optimizer_config"** (*dict[str, Any]*): Configuration of the optimizer.
                * **"loss_config"** (*dict[str, Any]*): Configuration of the loss function.
                * **"num_treaning_iterations"** (*int*): Number of training iterations (epochs).
        """
        self.change_device("CPU")
        node_configs = []
        for layer in self._LAYERS:
            node_configs.append(layer.recunstruct_layer_config())

        config = {
            'nodes_structure': self._NODE_STRUCTURE,
            'detailed_activations': node_configs,
            'learning_rate': self._LEARNING_RATE,
            'learning_mode': self._LEAR_MODE,
            'training_mode': self.training_mode,
            'batch_size': self._BATCH_SIZE
        }

        config.update({'initializer_config': self._INITIALIZER.get_config()})
        config['optimizer_config'] = self._UPDATE_METHOD.get_config()
        config["loss_config"] = self._LOSS_FUNCTION.get_config()
        config['num_treaning_iterations'] = self.num_treaning_iterations
        return config


class FlexibleNN(ConfigurableNN):
    """
    Intermediate class for creating dense neural networks with flexible activation configurations.
    Child class of :obj:`~HeteroSymNN.Core.Nets.neural_nets.ConfigurableNN`.
    
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
    learning_mode : str, optional
        Learning mode of the network. Currently only "Static" is supported., by default "Static"
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
            raise ValueError(
                f"Formato inválido para la activación: {config_item}.\n"+
                f"Se esperaba 'str' o 'tuple[str, dict[str, float]]'.\n"+
                "Ejemplos: 'relu', ('mish', {'beta': 1.0})"
            )
        
    def _expand_to_detailed(self, num_layers: int, nodes_per_layer: list[int], layer_configs: list[FlexibleNodeConfig]) -> list[list[NodeConfig]]:
        """
        Internal method to expand flexible layer configurations into detailed node configurations for each layer.
        
        Child class of :obj:`~HeteroSymNN.Core.Nets.neural_nets.FlexibleNN`.
        
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
    

class SimpleNN(FlexibleNN):
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
    learning_mode : str, optional
        Learning mode of the network. Currently only "Static" is supported., by default "Static"
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