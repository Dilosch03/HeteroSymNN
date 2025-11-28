import numpy as np
from typing import Literal
import warnings

from ..Backend import hardware as HW
from ..JIT.compiler import SymbolicJITCompiler

class Loss:
    def _change_COMPUTATIONAL_METHOD(self,new_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],gpu_id:int = None):
        pass
    def set_gpu_id(self,new_id:int):
        pass
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError

    def get_config(self):
        return {"class_name": self.__class__.__name__}

class FlexibleLoss(Loss):
    """
    Función de pérdida JIT-compilada.
    Acepta una expresión matemática (string) y constantes opcionales.
    """
    def __init__(self, loss_expression: str = "(y_pred - y_true)**2", constants: dict[str, float] = None,
                 computational_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None,gpu_id:int = 0):
        self.loss_expression = loss_expression
        self.constants = constants or {}
        self.COMPUTATIONAL_METHOD = HW.DEFAULT_COMPUTE_METHOD
        self.GPU_ID = gpu_id

        if (computational_method != None):
            if ((computational_method == "GPU_CUDA") and not(HW.GPU_ENABLED)):
                if (HW.WARNINGS_STRICT_MODE):
                    raise RuntimeError("Se intento cambiar al metodo de GPU_CUDA cuando no se tiene una gpu valida.")
                else:
                    warnings.warn("Se intento cambiar al metodo de GPU_CUDA cuando no se tiene una gpu valida."+"Intantando con el metodo CPU_JIT")
                    computational_method = "CPU_JIT"

            if ((computational_method == "CPU_JIT")and not(HW.CPP_JIT_ENABLED)):
                if (HW.WARNINGS_STRICT_MODE):
                    raise RuntimeError("Se intento cambiar al metodo de CPU_JIT cuando no se tiene un compilador de c++ valido.")
                else:
                    warnings.warn("Se intento cambiar al metodo de GPU_CUDA cuando no se tiene un compilador de c++ valido."+"Cambiando al metodo CPU_PYTHON")
                    computational_method = "CPU_PYTHON"

            self.COMPUTATIONAL_METHOD = computational_method

        
        self.compiler = SymbolicJITCompiler(
            configs=[(loss_expression, self.constants)],
            calculation_method=self.COMPUTATIONAL_METHOD,
            device_id=self.GPU_ID, 
            mode="loss"
        )

    @property
    def _be(self):
        """Propiedad para obtener el backend de cálculo actual (numpy o cupy)"""
        if (("GPU" in self.COMPUTATIONAL_METHOD) and HW.GPU_ENABLED):
            return HW.cp
        return np
    
    def set_gpu_id(self,new_id:int):
        if (new_id != self.GPU_ID):
            self.compiler.set_gpu_id(new_id)

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
        
        if (new_method != self.COMPUTATIONAL_METHOD):
            self.COMPUTATIONAL_METHOD = new_method
            self.GPU_ID = gpu_id

            self.COMPUTATIONAL_METHOD = self.compiler._change_method(self.COMPUTATIONAL_METHOD,self.GPU_ID)
        return self.COMPUTATIONAL_METHOD

    def forward(self, y_pred, y_true):
        # Aseguramos buffers del tamaño correcto en el dispositivo correcto
        loss_vec = self._be.zeros_like(y_pred)
        self.compiler.forward_kernel(y_pred, y_true, loss_vec)
        return self._be.mean(loss_vec) 

    def backward(self, y_pred, y_true):
        grad_vec = self._be.zeros_like(y_pred)
        self.compiler.backward_kernel(y_pred, y_true, grad_vec)
        # Normalización del gradiente para Mean reduction: (2 / N) o (1 / N) dependiendo de la convención.
        # Para MSE puro la derivada de sum((y-t)^2)/N es 2(y-t)/N.
        return grad_vec * (2.0 / y_pred.size) 

    def get_config(self):
        return {
            "class_name": "FlexibleLoss",
            "loss_expression": self.loss_expression,
            "constants": self.constants
        }

# --- Atajos Comunes ---

class MSELoss(FlexibleLoss):
    def __init__(self,computational_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None,gpu_id:int = 0):
        super().__init__("mse",computational_method=computational_method,gpu_id=gpu_id) 

class MAELoss(FlexibleLoss):
    def __init__(selfm,computational_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None,gpu_id:int = 0):
        super().__init__("mae",computational_method=computational_method,gpu_id=gpu_id)

# Ejemplo de Loss con Estado (Hiperparámetros)
class HuberLoss(FlexibleLoss):
    def __init__(self, delta=1.0,computational_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None,gpu_id:int = 0):
        # Usamos la definición de Huber que está en COMMON_FORMULAS o la definimos aquí
        # Nota: COMMON_FORMULAS['huber'] usa 1.0 hardcodeado. 
        # Para usar el delta dinámico, reescribimos la fórmula usando la variable 'delta'
        formula = "Piecewise((0.5 * (y_pred - y_true)**2, Abs(y_pred - y_true) <= delta), (delta * (Abs(y_pred - y_true) - 0.5 * delta), True))"
        super().__init__(formula, constants={"delta": delta},computational_method=computational_method,gpu_id=gpu_id)

class BinaryCrossEntropy(FlexibleLoss):
    def __init__(self,computational_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None,gpu_id:int = 0):
        super().__init__("bce",computational_method=computational_method,gpu_id=gpu_id)