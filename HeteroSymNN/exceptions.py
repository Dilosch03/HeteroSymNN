
class HeteroSymNNError(Exception):
    """
    Base class for all HeteroSymNN exceptions.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class BackendError(HeteroSymNNError):
    """
    Base for backend-related errors.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class JITError(HeteroSymNNError):
    """
    Base for JIT compilation errors.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class ConfigError(HeteroSymNNError):
    """
    Base for configuration errors.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class WrapperError(HeteroSymNNError):
    """
    Base for wrapper-related errors.
    """
    pass

class RuntimeStateError(HeteroSymNNError):
    """
    Exception raised when a method is called in an invalid execution state or order 
    (e.g., calling a backward pass before a forward pass).
    """


class MethodMigrationError(BackendError):
    """
    Exception raised when a computational method migration fails.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class ShapeMismatchError(HeteroSymNNError):
    """
    Exception raised when there is a mismatch in the shapes of the reciving data.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class BackendNotAvailableError(BackendError):
    """
    Exception raised when a backend is not available.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class InvalidDeviceIDError(BackendError):
    """
    Exception raised when an invalid device ID is provided.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class ResourceAllocationError(BackendError):
    """
    Exception raised when the hardware (specifically GPU VRAM) runs out of memory. 
    Usually indicates the batch size is too large or the network is too deep for the current device.
    """

class BackendDataTypeError(BackendError):
    """
    Exception raised when a tensor's data type (e.g., float64, int32) is incompatible 
    with the current backend's expected precision (e.g., float32), 
    preventing C++ JIT or CUDA kernel execution.
    """

class JITCompilationError(JITError):
    """
    Exception raised when JIT compilation fails.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class FormulaParsingError(JITError):
    """
    Exception raised when a formula parsing fails.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class NetworkStructureError(ConfigError):
    """
    Exception raised when there is an issue with the network structure.
    Invalid number of layer or activation mismaches.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class LayerConfigurationError(ConfigError):
    """
    Exception raised when there is an issue with the layer configuration.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class PathError(ConfigError):
    """
    Exception raised when there is an issue with the path.
    """
    pass

class TrainingError(WrapperError):
    """
    Exception raised when an error occurs during the training process within a wrapper.
    """
    pass

class LoadingError(WrapperError):
    """
    Exception raised when an error occurs during the loading process of models.
    """
    pass

class SavingError(WrapperError):
    """
    Exception raised when an error occurs during the saving process of models.
    """
    pass

class DataTypeError(HeteroSymNNError):
    """
    HeteroSymNN exception for type errors.
    """
    pass


class HeteroSymNNWarnings(UserWarning):
    """
    Base class for all HeteroSymNN warnings.
    """
    pass

class HardwareWarning(HeteroSymNNWarnings):
    """
    Warning raised when there is a hardware-related issue that doesn't prevent execution.
    """
    pass

class PerformanceWarning(HeteroSymNNWarnings):
    """
    Warning raised when a configuration might lead to suboptimal performance.
    """
    pass

class CompilationWarning(HeteroSymNNWarnings):
    """
    Warning raised when there was a problem with the creation of the kernels.
    """
    pass

class PathWarning(HeteroSymNNWarnings):
    """
    Warning where there is a problem with a path.
    """
    pass

class ShapeWarning(HeteroSymNNWarnings):
    """
    Warning raised when there is a mismatch in the shapes of the receiving data.
    """
    pass

class LoadingWarning(HeteroSymNNWarnings):
    """
    Warning raised when an issue occurs during the loading process that is not fatal.
    """
    pass
