import numpy as np
from typing import Literal
import warnings

from ..Backend import hardware as HW

class Optimizer:
    def __init__(self, learning_rate: float = None, computational_device:Literal["GPU", "CPU"]=None, device_id: int = None):
        self.learning_rate = learning_rate
        self.DEVICE_ID = device_id
        self.CURRENT_DEVICE = "CPU"
        self.COMPUTACIONAL_DEVICE = HW.DEFAULT_COMPUTE_METHOD.split("_")[0]
        self.be = HW.be 
        self._ASNUMPY = HW.asnumpy

        if (computational_device != None):
            self.COMPUTACIONAL_DEVICE = computational_device
            if ((computational_device == "GPU") and not (HW.GPU_ENABLED)):
                if (HW.WARNINGS_STRICT_MODE):
                    raise ValueError("Intentando definir como dispositivo computacional la GPU cuando esta no esta disponible.")
                else:
                    warnings.warn("Intentando definir como dispositivo computacional la GPU cuando esta no esta disponible."+"Cambiando el dispositivo computacional a la CPU ")
                    self.COMPUTACIONAL_DEVICE = "CPU"

            if (self.COMPUTACIONAL_DEVICE == "GPU"):
                self.be = HW.cp
                self._ASNUMPY = HW.cp.asnumpy
            else:
                self.be = np
                self._ASNUMPY = np.array
            
        self._setup_kernels()

    def _refresh_parameters(self, vector_format):
        raise NotImplementedError

    def _setup_kernels(self):
        pass

    def _change_COMPUTACIONAL_DEVICE(self, device:Literal["GPU","CPU"], device_id: int = None):
        if not(device in ["GPU","CPU"]):
            raise ValueError("Se paso como device algo que no es GPU o CPU.")
        
        if ((device == "GPU") and (not(HW.GPU_ENABLED))):
            if (HW.WARNINGS_STRICT_MODE):
                raise ValueError("Intentando definir como dispositivo computacional la GPU cuando esta no esta disponible.")
            else:
                warnings.warn("Intentando definir como dispositivo computacional la GPU cuando esta no esta disponible."+"Cambiando el dispositivo computacional a la CPU ")
                device = "CPU"

        if ((device == "GPU")and(self.COMPUTACIONAL_DEVICE == "CPU")):
            if(HW.WARNINGS_STRICT_MODE):
                raise RuntimeError("Se intento cambar a la gpu cuando se tiene como dispositivo computacional la cpu.")
            else:
                warnings.warn("Se intento cambar a la gpu cuando se tiene como dispositivo computacional la cpu. Se ignoro la peticion por seguridad")
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
        if (new_id >= HW.NUM_GPUS):
            raise ValueError("")

        if (new_id != self.DEVICE_ID):
            self.DEVICE_ID = new_id
            if (self.CURRENT_DEVICE == "GPU"):
                with HW.be.cuda.Device(self.DEVICE_ID):
                    self._refresh_parameters(HW.cp.array)
    
    def _to_device(self, device: Literal["GPU", "CPU"]):
        if not(device in ["GPU","CPU"]):
            raise ValueError("Se paso como device algo que no es GPU o CPU.")
        
        if ((device == "GPU")and(self.COMPUTACIONAL_DEVICE == "CPU")):
            if (HW.WARNINGS_STRICT_MODE):
                raise ValueError("Se intento cambiar a la GPU cuando se habia definido el dispositivo computacional como CPU")
            else:
                warnings.warn("Se intento cambiar a la GPU cuando se habia definido el dispositivo computacional como CPU."+"Ingorando peticion por seguridad.")
                device = "CPU"

        if (device != self.CURRENT_DEVICE):
            self.CURRENT_DEVICE = device
            
            if device == "GPU":
                with HW.be.cuda.Device(self.DEVICE_ID):
                    self._refresh_parameters(self.be.array)
            else:
                self._refresh_parameters(self._ASNUMPY)

    def step(self, layers: list, inputs):
        raise NotImplementedError

    def get_state(self):
        self._to_device("CPU")
        return {}

    def set_state(self, state, be):
        self._to_device("CPU")
        pass

    def get_config(self):
        self._to_device("CPU")
        return {'class_name': self.__class__.__name__, 'learning_rate': self.learning_rate}


class SgdOptimizer(Optimizer):
    _kernel_weights = None
    _kernel_bias = None

    def __init__(self, learning_rate: float = None,computational_device:Literal["GPU", "CPU"]=None, device_id: int = None):
        if (learning_rate is None):
            learning_rate = 0.01
        super().__init__(learning_rate,computational_device,device_id)

    def _refresh_parameters(self, vector_format):
        pass
    
    def _setup_kernels(self):
        if (HW.GPU_ENABLED):
            if (SgdOptimizer._kernel_weights is None):
                SgdOptimizer._kernel_weights = HW.cp.ElementwiseKernel(
                    'T grad, T lr, T mask',
                    'T param',
                    'param -= lr * grad * mask',
                    'sgd_weights_kernel'
                )
            if (SgdOptimizer._kernel_bias is None):
                SgdOptimizer._kernel_bias = HW.cp.ElementwiseKernel(
                    'T grad, T lr',
                    'T param',
                    'param -= lr * grad',
                    'sgd_bias_kernel'
                )

    def step(self, layers: list, inputs):
        self._to_device(self.COMPUTACIONAL_DEVICE)        
        prev_a = inputs

        for layer in layers:
            batch_size = layer.delta.shape[1]
            grad_b = self.be.mean(layer.delta, axis=1, keepdims=True)
            grad_w = self.be.dot(layer.delta, prev_a.T) / batch_size

            grad_w_masked = grad_w * layer.connection_mask

            if (self.CURRENT_DEVICE == "GPU"):
                SgdOptimizer._kernel_weights(grad_w, float(self.learning_rate), layer.connection_mask, layer.weights)
                SgdOptimizer._kernel_bias(grad_b, float(self.learning_rate), layer.biases)
            else:
                grad_w_masked = grad_w * layer.connection_mask
                layer.weights -= self.learning_rate * grad_w_masked
                layer.biases -= self.learning_rate * grad_b

            prev_a = layer.a


class AdamOptimizer(Optimizer):
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
            
        self.m = []
        self.v = []
        for layer in layers:

            m_w = self.be.zeros_like(layer.weights)
            v_w = self.be.zeros_like(layer.weights)
            
            m_b = self.be.zeros_like(layer.biases)
            v_b = self.be.zeros_like(layer.biases)
            
            self.m.append({'w': m_w, 'b': m_b})
            self.v.append({'w': v_w, 'b': v_b})

    def _refresh_parameters(self, vector_format):
        if self.m is None or self.v is None:
            return

        new_m = []
        new_v = []
        
        for i in range(len(self.m)):
            new_m_layer = {
                'w': vector_format(self.m[i]['w']),
                'b': vector_format(self.m[i]['b'])
            }
            new_v_layer = {
                'w': vector_format(self.v[i]['w']),
                'b': vector_format(self.v[i]['b'])
            }
            new_m.append(new_m_layer)
            new_v.append(new_v_layer)
            
        self.m = new_m
        self.v = new_v

    def step(self, layers: list, inputs):
        self._to_device(self.COMPUTACIONAL_DEVICE)
        if self.learning_rate is None:
            self.learning_rate = 0.001

        if self.m is None:
            self._initialize_state(layers)

        self.t += 1
        prev_a = inputs

        t_pow_beta1 = self.beta1 ** self.t
        t_pow_beta2 = self.beta2 ** self.t

        for i, layer in enumerate(layers):
            batch_size = layer.delta.shape[1]

            grad_b = self.be.mean(layer.delta, axis=1, keepdims=True)
            grad_w = self.be.dot(layer.delta, prev_a.T) / batch_size

            m_t = self.m[i]
            v_t = self.v[i]

            if (self.CURRENT_DEVICE == "GPU"):
                # Weights
                AdamOptimizer._fused_kernel(
                    grad_w, float(self.learning_rate), float(self.beta1), float(self.beta2), float(self.epsilon), 
                    float(t_pow_beta1), float(t_pow_beta2), layer.connection_mask,
                    layer.weights, m_t['w'], v_t['w'])
                # Biases
                AdamOptimizer._fused_kernel(
                    grad_b, float(self.learning_rate), float(self.beta1), float(self.beta2), float(self.epsilon),
                    float(t_pow_beta1), float(t_pow_beta2), 1.0,
                    layer.biases, m_t['b'], v_t['b']
                )

            else:
                grad_w_masked = grad_w * layer.connection_mask

                m_t['w'] = self.beta1 * m_t['w'] + (1 - self.beta1) * grad_w_masked
                v_t['w'] = self.beta2 * v_t['w'] + (1 - self.beta2) * (grad_w_masked ** 2)
                m_w_hat = m_t['w'] / (1 - t_pow_beta1)
                v_w_hat = v_t['w'] / (1 - t_pow_beta2)
                layer.weights -= self.learning_rate * m_w_hat / (self.be.sqrt(v_w_hat) + self.epsilon)

                m_t['b'] = self.beta1 * m_t['b'] + (1 - self.beta1) * grad_b
                v_t['b'] = self.beta2 * v_t['b'] + (1 - self.beta2) * (grad_b ** 2)
                m_b_hat = m_t['b'] / (1 - t_pow_beta1)
                v_b_hat = v_t['b'] / (1 - t_pow_beta2)
                layer.biases -= self.learning_rate * m_b_hat / (self.be.sqrt(v_b_hat) + self.epsilon)


            prev_a = layer.a

    def get_state(self):
        super().get_state()
        if self.m is None:
            return {'t': self.t, 'm': None, 'v': None}
        
        m_np = [{'w': self._ASNUMPY(lay['w']), 'b': self._ASNUMPY(lay['b'])} for lay in self.m]
        v_np = [{'w': self._ASNUMPY(lay['w']), 'b': self._ASNUMPY(lay['b'])} for lay in self.v]

        return {'t': self.t, 'm': m_np, 'v': v_np}

    def set_state(self, state, be):
        super().set_state()
        self.t = state.get('t', 0)
        m_data = state.get('m')
        v_data = state.get('v')

        if m_data is None or v_data is None:
            self.m = None
            self.v = None
            return

        self.m = [{'w': be.array(lay['w']), 'b': be.array(lay['b'])} for lay in m_data]
        self.v = [{'w': be.array(lay['w']), 'b': be.array(lay['b'])} for lay in v_data]

    def get_config(self):
        config = super().get_config()
        config.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        })
        return config