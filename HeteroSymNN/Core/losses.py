from __future__ import annotations
import numpy as np
from typing import Literal
import warnings

from ..Backend import hardware as HW
from ..JIT.compiler import SymbolicJITCompiler
from ..types import BackendArray
from ..config import settings
from ..exceptions import BackendNotAvailableWarning, HardwareError
from ..error_handlers import clean_traceback

__all__ = [
    "Loss", "FlexibleLoss",
    "MSELoss", "MAELoss", "HuberLoss", "BinaryCrossEntropy",
]


class Loss:
    """
    Base class for all loss functions.
    """
    def _change_COMPUTATIONAL_METHOD(self,new_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],gpu_id:int = None):
        """
        Internal method to change the computational backend.

        This method must be implemented by subclasses to handle the transfer of internal data 
        (e.g., constants) between GPU and CPU memory spaces when switching devices. It must also 
        update the execution strategy (distinguishing between ``CPU_JIT`` and ``CPU_PYTHON``) 
        to ensure the correct processing path is used.
        """
        pass
    def set_gpu_id(self,new_id:int):
        """
        Sets the GPU ID for computation.

        This method must be implemented by subclasses to update the GPU ID used for internal calculations.

        Parameters
        ----------
        new_id : int
            The new GPU ID.
        """
        pass
    def forward(self, y_pred:BackendArray, y_true:BackendArray)->BackendArray:
        """
        Computes the loss value.

        This method must be implemented by subclasses to define the forward pass of the loss function.

        Parameters
        ----------
        y_pred : :type:`~HeteroSymNN.types.BackendArray`
            The predicted values.
        y_true : :type:`~HeteroSymNN.types.BackendArray`
            The ground truth values.
        
        Returns
        -------
        :type:`~HeteroSymNN.types.BackendArray`
            The loss value.
        """
        raise NotImplementedError

    def backward(self, y_pred:BackendArray, y_true:BackendArray)->BackendArray:
        """
        Computes the gradient of the loss.

        This method must be implemented by subclasses to define the backward pass (gradient computation) of the loss function.

        Parameters
        ----------
        y_pred : :type:`~HeteroSymNN.types.BackendArray`
            The predicted values.
        y_true : :type:`~HeteroSymNN.types.BackendArray`
            The ground truth values.

        Returns
        -------
        :type:`~HeteroSymNN.types.BackendArray`
            The gradient of the loss.
        """
        raise NotImplementedError

    def get_config(self):
        """
        Returns the configuration of the loss instance.

        This method should be implemented by subclasses to return a dictionary containing the parameters necessary 
        to reconstruct the loss function and its internal state.
        """
        return {"class_name": self.__class__.__name__}

class FlexibleLoss(Loss):
    """
    Base class for loss functions that use the JIT compiler.
    
    Parameters
    ----------
    loss_expression : str
        The definition of the loss function. This can be:
        
        1. A valid mathematical Python expression using ``y_pred`` and ``y_true`` (e.g., ``"(y_pred - y_true)**2"``).
        2. A case-insensitive key from the predefined ``COMMON_FORMULAS`` (e.g., ``"mse"``, ``"bce"``, ``"huber"``).

    constants : dict[str, float], optional
        Dictionary of the non-standard constants that are in the loss function if any, by default None.
    computational_method : Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"], optional
        The computational method that is going to be used, by default None.
    gpu_id : int, optional
        The GPU ID that is going to be used if method is "GPU_CUDA", by default 0.

    Examples
    --------
    >>> from HeteroSymNN.Core.Nets import MLP
    >>> from HeteroSymNN.Core.losses import FlexibleLoss
    >>> NN = MLP(nodes_structure=[4, 6, 3],
    ...        activation="sigmoid",
    ...        loss_function=FlexibleLoss("abs(y_pred - y_true)"))
    """
    @clean_traceback
    def __init__(self, loss_expression: str = "(y_pred - y_true)**2", constants: dict[str, float] = None,
                 computational_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None,gpu_id:int = 0):
        self._loss_expression = loss_expression
        self._constants = constants or {}
        self._COMPUTATIONAL_METHOD = settings.default_compute_method
        
        if (0>=gpu_id >= HW.NUM_GPUS):
            raise HardwareError("The GPU requested form Id is not in the list of available GPUs.")
        self._GPU_ID = gpu_id
        self._be = settings.default_manager
        self._asnumpy = settings.default_asnumpy

        if not(computational_method is None):
            computational_method = computational_method.upper()
            try_method = computational_method
            msg_extra = ""
            if not(computational_method in ["GPU_CUDA","CPU_JIT","CPU_PYTHON"]):
                raise ValueError("Tried to change the computational method to something that isn't GPU_CUDA, CPU_JIT or CPU_PYTHON")

            if ((computational_method == "GPU_CUDA") and not(computational_method in settings.available_methods)):
                msg_extra = ", but no GPU is available."
                computational_method = "CPU_PYTHON"

            if ((computational_method == "CPU_JIT")):
                    msg_extra = ", but currently is not available"
                    computational_method = "CPU_PYTHON"

            if (try_method != computational_method):
                    warnings.warn(f"Tried to change to use '{try_method}'{msg_extra}. {computational_method} is required",BackendNotAvailableWarning,stacklevel=2)

            if (computational_method == "GPU_CUDA"):
                self._be = HW.cp
                self._asnumpy = HW.cp.asnumpy
            else:
                self._be = np
                self._asnumpy = np.array

            self._COMPUTATIONAL_METHOD = computational_method

        self.set_constants(self._constants)

        self._compiler = SymbolicJITCompiler(
            configs=[(loss_expression, self._constants)],
            calculation_method=self._COMPUTATIONAL_METHOD,
            device_id=self._GPU_ID, 
            mode="loss"
        )
    
    @property
    def LOSS_EXPRESSION(self)->str:
        """
        Expression or name of the loss function used.
        
        Returns
        -------
        str
        """
        return self._loss_expression
    
    @property
    def COMPUTATIONAL_METHOD(self)->Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]:
        """
        Current computational method used.
        
        Returns
        -------
        Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]
        """
        return self._COMPUTATIONAL_METHOD
    
    @property
    def GPU_ID(self)->int:
        """
        Current GPU ID used.
        
        Returns
        -------
        int
        """
        return self._GPU_ID

    @property
    def COMPILER(self)->SymbolicJITCompiler:
        """
        Compiler class that is used for the symbolic loss function.
        
        Returns
        -------
        SymbolicJITCompiler
        """
        return self._compiler
    
    def set_constants(self,constants:dict[str,float])->None:
        """
        Method to set the constants of the loss function using a dictionary of the string of the constant and its new value.
        
        Parameters
        ----------
        constants : dict[str, float]
        """
        temp = []
        for key in sorted(constants.keys()):
            temp.append(constants[key])

        if (len(temp) == 0):
            temp.append(0.0)

        if (self._COMPUTATIONAL_METHOD.split("_")[0] == "GPU"):
            with HW.be.cuda.Device(self._GPU_ID):
                self.arr_constants = self._be.array(temp,dtype=settings.default_dtype)
        else:
            self.arr_constants = self._be.array(temp,dtype=settings.default_dtype)

    def set_gpu_id(self,new_id:int)->None:
        """
        Method to set a new GPU to do the computation.

        Parameters
        ----------
        new_id : int
        """
        if (new_id != self._GPU_ID):
            self._compiler.set_gpu_id(new_id)
            self._GPU_ID = new_id

    def _change_COMPUTATIONAL_METHOD(self,new_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],gpu_id:int = None)->Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]:
        """
        Method to change the computational method used.

        This will force a kernel recompilation and the movement of the location of the constants.

        Parameters
        ----------
        new_method : Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]
            New computational method to set.
        gpu_id : int, optional
            GPU ID to use if the new method is "GPU_CUDA". If not provided, the current GPU ID already saved will be used., by default None.
        
        Raises
        ------
        ValueError
            If the new method is not one of "GPU_CUDA", "CPU_JIT", or "CPU_PYTHON".
        BackendNotAvailableError
            If trying to set "GPU_CUDA" without a valid GPU or "CPU_JIT" without a valid C++ compiler when strict warnings mode is enabled. If not enabled, it will fallback to the next available method and throw a warning.
        
        Returns
        -------
        Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"]
            New computational method it was able to be set. In case that there was an error in the recompilation and `HeteroSymNN.Backend.hardware.WARNINGS_STRICT_MODE` is set to false the method that was requested will differ with the returned method.
        """
        new_method = new_method.upper()
        try_method = new_method
        msg_extra = ""
        if not(new_method in ["GPU_CUDA","CPU_JIT","CPU_PYTHON"]):
            raise ValueError("tried to change the computational method to something that isn't GPU_CUDA, CPU_JIT or CPU_PYTHON")
        
        if (gpu_id == None):
            gpu_id = self._GPU_ID
        else:
            if (0>=gpu_id >= HW.NUM_GPUS):
                raise HardwareError("The GPU requested form Id is not in the list of available GPUs.")

        if ((new_method == "GPU_CUDA") and not(new_method in settings.available_methods)):
            msg_extra = ", but no GPU is available."
            new_method = "CPU_PYTHON"

        if ((new_method == "CPU_JIT")):
                msg_extra = ", but currently is not available"
                new_method = "CPU_PYTHON"

        if (try_method != new_method):
                warnings.warn(f"Tried to change to use '{try_method}'{msg_extra}. {new_method} is required",BackendNotAvailableWarning,stacklevel=2)

        
        if (new_method != self._COMPUTATIONAL_METHOD):
            self._COMPUTATIONAL_METHOD = new_method
            self._GPU_ID = gpu_id
            temp = self._asnumpy(self.arr_constants)
            
            self._COMPUTATIONAL_METHOD = self._compiler._change_method(self._COMPUTATIONAL_METHOD,self._GPU_ID)
            if (self._COMPUTATIONAL_METHOD == "GPU_CUDA"):
                self._asnumpy = HW.cp.asnumpy
                self._be = HW.cp
            else:
                self._asnumpy = np.array
                self._be = np

            self.arr_constants = self._be.array(temp,dtype=settings.default_dtype)

        return self._COMPUTATIONAL_METHOD

    @clean_traceback
    def forward(self, y_pred:BackendArray, y_true:BackendArray):
        """
        Forward pass of the loss function.

        Parameters
        ----------
        y_pred : :type:`~HeteroSymNN.types.BackendArray`
            Predicted values.
        y_true : :type:`~HeteroSymNN.types.BackendArray`
            True values.

        Returns
        -------
        :type:`~HeteroSymNN.types.BackendArray`
            Loss value.
        """
        loss_vec = self._be.zeros_like(y_pred,dtype=settings.default_dtype)
        self._compiler.forward_kernel(y_pred, y_true, loss_vec,self.arr_constants)
        return self._be.mean(loss_vec) 

    @clean_traceback
    def backward(self, y_pred:BackendArray, y_true:BackendArray):
        """
        Backward pass of the loss function.

        Parameters
        ----------
        y_pred : :type:`~HeteroSymNN.types.BackendArray`
            Predicted values.
        y_true : :type:`~HeteroSymNN.types.BackendArray`
            True values.

        Returns
        -------
        :type:`~HeteroSymNN.types.BackendArray`
            Gradient of the loss with respect to the predicted values.
        """
        grad_vec = self._be.zeros_like(y_pred,dtype=settings.default_dtype)
        self._compiler.backward_kernel(y_pred, y_true, grad_vec,self.arr_constants)
        return grad_vec * (1.0 / y_pred.size) 
    
    def get_constants(self)->dict[str,float]:
        """
        Method to get the constants values of the loss function if it has any.
        
        Returns
        -------
        dict[str,float]
        """
        temp_array = self.arr_constants.copy()
        if (self._COMPUTATIONAL_METHOD.split("_")[0] == "GPU"):
            temp_array = self._be.asnumpy(temp_array)
        
        reconstructed_constants = {}
        for i,key in enumerate(sorted(self._constants.keys())):
            reconstructed_constants[key] = float(temp_array[i])

        return reconstructed_constants
    
    def get_config(self):
        """
        Returns the configuration of the loss instance.

        This method returns a dictionary containing the parameters necessary 
        to reconstruct the loss function and its internal state.
        
        Returns
        -------
        dict[str,any]
            A dictionary containing the class name, loss expression, and constants.

            **class_name** : str
                The name of the class.
            **loss_expression** : str
                The definition of the loss function.
            **constants** : dict[str,float]
                dictionary of the non-standard constants that are in the loss function if any.
        """
        return {
            "class_name": "FlexibleLoss",
            "loss_expression": self._loss_expression,
            "constants": self.get_constants()
        }


class MSELoss(FlexibleLoss):
    """
    High level class for calling for a MSE loss function.
    
    Parameters
    ----------
    computational_method : Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"], optional
        The computational method that is going to be used, by default None.
    gpu_id : int, optional
        The GPU ID that is going to be used if method is "GPU_CUDA", by default 0.
    """
    @clean_traceback
    def __init__(self,computational_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None,gpu_id:int = 0):
        super().__init__("mse",computational_method=computational_method,gpu_id=gpu_id) 

class MAELoss(FlexibleLoss):
    """
    High level class for calling for a MAE loss function.
    
    Parameters
    ----------
    computational_method : Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"], optional
        The computational method that is going to be used, by default None.
    gpu_id : int, optional
        The GPU ID that is going to be used if method is "GPU_CUDA", by default 0.
    """
    @clean_traceback
    def __init__(self,computational_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None,gpu_id:int = 0):
        super().__init__("mae",computational_method=computational_method,gpu_id=gpu_id)


class HuberLoss(FlexibleLoss):
    """
    High level class for calling for a Huber loss function.
    
    Parameters
    ----------
    delta: float
        Value of the constant Delta
    computational_method : Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"], optional
        The computational method that is going to be used, by default None.
    gpu_id : int, optional
        The GPU ID that is going to be used if method is "GPU_CUDA", by default 0.
    """
    @clean_traceback
    def __init__(self, delta:float=1.0,computational_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None,gpu_id:int = 0):
        formula = "Piecewise((0.5 * (y_pred - y_true)**2, Abs(y_pred - y_true) <= delta), (delta * (Abs(y_pred - y_true) - 0.5 * delta), True))"
        super().__init__(formula, constants={"delta": delta},computational_method=computational_method,gpu_id=gpu_id)

class BinaryCrossEntropy(FlexibleLoss):
    """
    High level class for calling for a Binary Cross Entropy loss function.
    
    Parameters
    ----------
    computational_method : Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"], optional
        The computational method that is going to be used, by default None.
    gpu_id : int, optional
        The GPU ID that is going to be used if method is "GPU_CUDA", by default 0.
    """
    @clean_traceback
    def __init__(self,computational_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"] = None,gpu_id:int = 0):
        super().__init__("bce",computational_method=computational_method,gpu_id=gpu_id)