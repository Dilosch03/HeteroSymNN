from __future__ import annotations
import numpy as np
from typing import Literal,Optional,Union
import warnings
import concurrent.futures

from ..Backend import hardware as HW
from ..exceptions import PerformanceWarning,BackendNotAvailableError,InvalidDeviceIDError
from ..config import settings
from ..types import BackendArray

class Optimizer:
    """
    Base class for all optimizers.
    
    Parameters
    ----------
    learning_rate : float, optional
        The learning rate for the optimizer.
    computational_device : Literal["GPU", "CPU"], optional
        The device where computations will be performed.
    device_id : int, optional
        The ID of the GPU to use if computational_device is "GPU".
    """
    def __init__(self, learning_rate: float = None, computational_device:Optional[Literal["GPU", "CPU"]]=None, device_id: Optional[int] = None):
        self.learning_rate = learning_rate
        self.DEVICE_ID = device_id
        self.CURRENT_DEVICE = "CPU"
        self.COMPUTACIONAL_DEVICE = settings.default_compute_method.split("_")[0]
        self.be = HW.be 
        self._ASNUMPY = HW.asnumpy
        self._thread_pool = None

        if (computational_device != None):
            self.COMPUTACIONAL_DEVICE = computational_device
            if ((computational_device == "GPU") and not (HW.GPU_ENABLED)):
                if (settings.warning_level == "error"):
                    raise BackendNotAvailableError("Trying to define the GPU as the computational device when there is no GPU available.")
                elif (settings.warning_level == "warn"):
                    warnings.warn("Trying to define the GPU as the computational device when there is no GPU available."+"Using the CPU as fallback.",PerformanceWarning,stacklevel=3)
                    self.COMPUTACIONAL_DEVICE = "CPU"

            if (self.COMPUTACIONAL_DEVICE == "GPU"):
                self.be = HW.cp
                self._ASNUMPY = HW.cp.asnumpy
            else:
                self.be = np
                self._ASNUMPY = np.array
            
        self._setup_kernels()

    def _refresh_parameters(self, vector_format):
        """
        Internal method to refresh internal parameters when changing devices or vector formats.

        This method must be implemented by subclasses to ensure that all internal state tensors 
        (e.g., momentum, velocity) are converted to the correct backend format (NumPy or CuPy) 
        provided by ``vector_format``.
        """
        raise NotImplementedError

    def _setup_kernels(self):
        """
        Internal method to setup CUDA kernels if needed.

        This method should be implemented by subclasses to compile or define any custom CUDA kernels 
        required for the optimizer when running on a GPU.
        """
        pass

    def _change_COMPUTACIONAL_DEVICE(self, device:Literal["GPU","CPU"], device_id: Optional[int] = None):
        """
        Internal method to change the computational device (CPU/GPU).
        
        Parameters
        ----------
        device : Literal["GPU", "CPU"]
            The new device to use.
        device_id : int, optional
            The GPU ID to use if device is "GPU".
        """
        if not(device in ["GPU","CPU"]):
            raise ValueError("Device not recognized. Expecting GPU or CPU.")
        
        if ((device == "GPU") and (not(HW.GPU_ENABLED))):
            if (settings.warning_level == "error"):
                raise BackendNotAvailableError("Trying to define the GPU as the computational device when there is no GPU available.")
            elif (settings.warning_level == "warn"):
                warnings.warn("Trying to define the GPU as the computational device when there is no GPU available."+"Using the CPU as fallback.",PerformanceWarning,stacklevel=3)
                device = "CPU"
        
        if (self.COMPUTACIONAL_DEVICE != device):
            self.COMPUTACIONAL_DEVICE = device
            if ((device == "GPU")and(HW.GPU_ENABLED)):
                self.be = HW.cp
                self._ASNUMPY = HW.cp.asnumpy
            else:
                self.be = np
                self._ASNUMPY = np.array

            if (device_id != None):
                self.DEVICE_ID = device_id

    def set_gpu_id(self,new_id:int):
        """
        Sets the GPU ID for the optimizer.
        
        Parameters
        ----------
        new_id : int
            The new GPU ID.
        """
        if (new_id >= HW.NUM_GPUS):
            raise InvalidDeviceIDError(f"ID given ({new_id}) is greater than the number of available GPUs ({HW.NUM_GPUS})")

        if (new_id != self.DEVICE_ID):
            self.DEVICE_ID = new_id
            if (self.CURRENT_DEVICE == "GPU"):
                with HW.be.cuda.Device(self.DEVICE_ID):
                    self._refresh_parameters(HW.cp.array)
    
    def _to_device(self, device: Literal["GPU", "CPU"]):
        """
        Internal method to move optimizer state to a specific device.
        
        Parameters
        ----------
        device : Literal["GPU", "CPU"]
            The target device.
        """
        if not(device in ["GPU","CPU"]):
            raise ValueError("Device not recognized. Expecting GPU or CPU.")
        
        if ((device == "GPU")and(self.COMPUTACIONAL_DEVICE == "CPU")):
            if (settings.warning_level == "error"):
                raise BackendNotAvailableError("Tried to send the paramters to the GPU when the CPU was set as the computational device.")
            elif (settings.warning_level == "warn"):
                warnings.warn("Tried to send the paramters to the GPU when the CPU was set as the computational device."+"Using the CPU as fallback for safety.",PerformanceWarning,stacklevel=3)
                device = "CPU"

        if (device != self.CURRENT_DEVICE):
            self.CURRENT_DEVICE = device
            
            if device == "GPU":
                with HW.be.cuda.Device(self.DEVICE_ID):
                    self._refresh_parameters(self.be.array)
            else:
                self._refresh_parameters(self._ASNUMPY)

    def _single_update(self, layer, param_name: str, param: BackendArray, grad: BackendArray, mask: Union[float, BackendArray] = 1.0):
        """
            Internal method to update a single parameter of a layer.

            This method must be implemented by subclasses to define the specific optimization logic 
                (e.g., SGD update, Adam update) applied to the layers.
            
            Parameter
            ---------
            layer : :obj:`~HeteroSymNN.Core.Nets.layers.Layer`
                The layer to update.
            param_name : str
                The name of the parameter to update.
            param : :obj:`~HeteroSymNN.types.BackendArray`
                The current values of the parameter.
            grad : :obj:`~HeteroSymNN.types.BackendArray`
                The gradient of the parameter.
        """
        raise NotImplementedError

        
    def step(self, layers: list):
        """
        Performs a single optimization step.

        Universal routing loop. Iterates over layers and their working dictionaries.

        Parameters
        ----------
        layers : list
            List of layers to update.
        """
        self._to_device(self.COMPUTACIONAL_DEVICE)
        
        if self.CURRENT_DEVICE == "GPU":
            for layer in layers:
                self._route_layer(layer)
        else:
            # The "Lazy Trapdoor": Creates the persistent pool exactly once.
            if self._thread_pool is None:
                self._thread_pool = concurrent.futures.ThreadPoolExecutor()
            
            # The "Hot Path": Dispatch work to awake threads.
            # list() acts as a barrier, forcing the main thread to wait for all parallel updates to finish.
            list(self._thread_pool.map(self._route_layer, layers))

    def _route_layer(self, layer):
        """Helper method to isolate the logic for a single thread/loop pass."""
        params = layer.working_parameters
        grads = layer.working_gradients
        masks = getattr(layer, 'working_masks', {}) 

        for param_name in params:
            p = params[param_name]
            g = grads[param_name]
            m = masks.get(param_name, 1.0)
            
            self._single_update(layer, param_name, p, g, m)

    def get_state(self):
        """
        Returns the internal state of the optimizer.
        
        This method should be implemented by subclasses to return a dictionary containing 
        the current internal state (e.g., iteration count, moving averages) for serialization.

        Returns
        -------
        dict
            Dictionary containing the optimizer state.
        """
        self._to_device("CPU")
        return {}

    def set_state(self, state, be):
        """
        Sets the internal state of the optimizer.
        
        This method should be implemented by subclasses to restore the internal state 
        from a provided dictionary.

        Parameters
        ----------
        state : dict
            The state dictionary to load.
        be : module
            The backend module (numpy or cupy) to use for creating arrays.
        """
        self._to_device("CPU")

    def get_config(self):
        """
        Returns the configuration of the optimizer.
        
        This method should be implemented by subclasses to return a dictionary containing 
        the configuration parameters necessary to reconstruct the optimizer instance.

        Returns
        -------
        dict
            Dictionary containing the configuration parameters.
        """
        self._to_device("CPU")
        return {'class_name': self.__class__.__name__, 'learning_rate': self.learning_rate}


class SgdOptimizer(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    
    Parameters
    ----------
    learning_rate : float, optional
        The learning rate. Defaults to 0.01.
    computational_device : Literal["GPU", "CPU"], optional
        The device where computations will be performed.
    device_id : int, optional
        The ID of the GPU to use if computational_device is "GPU".
    """
    _kernel_weights = None
    _kernel_bias = None

    def __init__(self, learning_rate: float = None,computational_device:Optional[Literal["GPU", "CPU"]]=None, device_id: Optional[int] = None):
        if (learning_rate is None):
            learning_rate = 0.01
        super().__init__(learning_rate,computational_device,device_id)

    def _refresh_parameters(self, vector_format):
        pass
    
    def _setup_kernels(self):
        if (HW.GPU_ENABLED):
            if (SgdOptimizer._kernel is None):
                SgdOptimizer._kernel = HW.cp.ElementwiseKernel(
                    'T grad, T lr, T mask',
                    'T param',
                    'param -= lr * grad * mask',
                    'sgd_universal_kernel'
                )

    def _single_update(self, layer, param_name: str, param: BackendArray, grad: BackendArray, mask: Union[float, BackendArray] = 1.0):
        """
            Internal method to update a single parameter of a layer using SGD.
            
            Parameter
            ---------
            layer : :obj:`~HeteroSymNN.Core.Nets.layers.Layer`
                The layer to update.
            param_name : str
                The name of the parameter to update.
            param : :obj:`~HeteroSymNN.types.BackendArray`
                The current values of the parameter.
            grad : :obj:`~HeteroSymNN.types.BackendArray`
                The gradient of the parameter.
        """
        if (self.CURRENT_DEVICE == "GPU"):
            SgdOptimizer._kernel_weights(grad, float(self.learning_rate), mask, param)
        else:
            param -= self.learning_rate * grad * mask


    def step(self, layers: list):
        """
        Performs a single optimization step using SGD.
        
        Parameters
        ----------
        layers : list
            List of layers to update.
        inputs : Any
            Input data.
        """
        super().step(layers)

class AdamOptimizer(Optimizer):
    """
    Adam optimizer.
    
    Parameters
    ----------
    learning_rate : float, optional
        The learning rate. Defaults to 0.001.
    computational_device : Literal["GPU", "CPU"], optional
        The device where computations will be performed.
    device_id : int, optional
        The ID of the GPU to use if computational_device is "GPU".
    beta1 : float, optional
        The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta2 : float, optional
        The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon : float, optional
        A small constant for numerical stability. Defaults to 1e-8.
    """
    _fused_kernel = None
    def __init__(self, learning_rate: float = None,computational_device:Literal["GPU", "CPU"]=None, device_id: int = None, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate,computational_device,device_id)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.t = 0
        self.m = None 
        self.v = None 

    def _setup_kernels(self):
        if  ((HW.GPU_ENABLED) and (AdamOptimizer._fused_kernel is None)):
            AdamOptimizer._fused_kernel = HW.cp.ElementwiseKernel(
                'T grad, T lr, T beta1, T beta2, T eps, T beta1_t, T beta2_t, T mask',
                'T param, T m, T v',
                '''
                T g = grad * mask;
                m = beta1 * m + (1.0 - beta1) * g;
                v = beta2 * v + (1.0 - beta2) * g * g;
                T m_hat = m / (1.0 - beta1_t);
                T v_hat = v / (1.0 - beta2_t);
                param -= lr * m_hat / (sqrt(v_hat) + eps);
                ''',
                'adam_fused_kernel'
            )

    def _initialize_state(self, layers: list):
        if self.learning_rate is None:
            self.learning_rate = 0.001
            
        self.m = {}
        self.v = {}
        # Track the structural order of layers for saving later
        self._layer_order = [] 

        for i, layer in enumerate(layers):
            layer_id = id(layer)
            self._layer_order.append(layer_id)
            
            # Check the Staging Area: Did we load state from disk?
            if hasattr(self, '_loaded_m') and self._loaded_m is not None:
                # Pull the saved parameters for this specific layer index
                saved_m = self._loaded_m[i]
                saved_v = self._loaded_v[i]
                
                # Dynamically load whatever keys exist ('weights', 'biases') and push to hardware
                self.m[layer_id] = {k: self.be.array(v) for k, v in saved_m.items()}
                self.v[layer_id] = {k: self.be.array(v) for k, v in saved_v.items()}
            else:
                # Standard initialization with zeros
                self.m[layer_id] = {k: self.be.zeros_like(p) for k, p in layer.working_parameters.items()}
                self.v[layer_id] = {k: self.be.zeros_like(p) for k, p in layer.working_parameters.items()}

        # Clean up the staging area so it doesn't re-trigger
        if hasattr(self, '_loaded_m'):
            self._loaded_m = None
            self._loaded_v = None

    def _refresh_parameters(self, vector_format):
        if ((getattr(self, 'm', None) is None) or (getattr(self, 'v', None) is None)):
            return

        new_m = {}
        new_v = {}
        

        for layer_id, layer_m in self.m.items():
            new_m[layer_id] = {
                param_name: vector_format(tensor) 
                for param_name, tensor in layer_m.items()
            }
            
        for layer_id, layer_v in self.v.items():
            new_v[layer_id] = {
                param_name: vector_format(tensor) 
                for param_name, tensor in layer_v.items()
            }
            
        self.m = new_m
        self.v = new_v
    
    def _single_update(self, layer, param_name: str, param: BackendArray, grad: BackendArray, mask: Union[float, BackendArray] = 1.0):
        """
            Internal method to update a single parameter of a layer using Adam.
            
            Parameter
            ---------
            layer : :obj:`~HeteroSymNN.Core.Nets.layers.Layer`
                The layer to update.
            param_name : str
                The name of the parameter to update.
            param : :obj:`~HeteroSymNN.types.BackendArray`
                The current values of the parameter.
            grad : :obj:`~HeteroSymNN.types.BackendArray`
                The gradient of the parameter.
        """
        m_t = self.m[id(layer)][param_name]
        v_t = self.v[id(layer)][param_name]

        if (self.CURRENT_DEVICE == "GPU"):
            AdamOptimizer._fused_kernel(
                grad, float(self.learning_rate), float(self.beta1), float(self.beta2), float(self.epsilon), 
                float(self.t_pow_beta1), float(self.t_pow_beta2), mask,
                param, m_t, v_t
            )
        else:
            grad_masked = grad * mask

            # Update momentums
            m_t = self.beta1 * m_t + (1 - self.beta1) * grad_masked
            v_t = self.beta2 * v_t + (1 - self.beta2) * (grad_masked ** 2)
            
            # Save updated states back to dictionaries
            self.m[id(layer)][param_name] = m_t
            self.v[id(layer)][param_name] = v_t
            
            # Bias correction
            m_hat = m_t / (1 - self.t_pow_beta1)
            v_hat = v_t / (1 - self.t_pow_beta2)
            
            # Apply gradients
            param -= self.learning_rate * m_hat / (self.be.sqrt(v_hat) + self.epsilon)

    def step(self, layers: list):
        """
        Performs a single optimization step using Adam.
        
        Parameters
        ----------
        layers : list
            List of layers to update.
        """
        if self.learning_rate is None:
            self.learning_rate = 0.001

        if self.m is None:
            self._initialize_state(layers)

        self.t += 1
        
        self.t_pow_beta1 = self.beta1 ** self.t
        self.t_pow_beta2 = self.beta2 ** self.t

        super().step(layers)
    
    def get_state(self):
        super().get_state()
        if not hasattr(self, 'm') or self.m is None:
            return {'t': getattr(self, 't', 0), 'm': None, 'v': None}
        
        m_list = []
        v_list = []
        
        # Use the structural order to guarantee alignment during load
        for layer_id in self._layer_order:
            layer_m = self.m[layer_id]
            layer_v = self.v[layer_id]
            
            # Dynamically convert all parameters back to pure NumPy for serialization
            m_list.append({k: self._ASNUMPY(v) for k, v in layer_m.items()})
            v_list.append({k: self._ASNUMPY(v) for k, v in layer_v.items()})

        return {'t': self.t, 'm': m_list, 'v': v_list}

    def set_state(self, state, be):
        super().set_state(state, be)
        self.t = state.get('t', 0)
        m_data = state.get('m')
        v_data = state.get('v')

        if m_data is None or v_data is None:
            self.m = None
            self.v = None
            return

        self._loaded_m = m_data
        self._loaded_v = v_data

    def get_config(self):
        config = super().get_config()
        config.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        })
        return config