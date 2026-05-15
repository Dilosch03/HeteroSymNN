__all__ = [
    # Base
    "HeteroSymNNError", "HeteroSymNNWarnings",
    # Backend
    "BackendError", "BackendWarning",
    "MethodMigrationError", "MethodMigrationWarning",
    "BackendNotAvailableError", "BackendNotAvailableWarning",
    "InvalidDeviceIDError", "InvalidDeviceIDWarning",
    "ResourceAllocationError", "ResourceAllocationWarning",
    "BackendDataTypeError", "BackendDataTypeWarning",
    "HardwareWarning",
    # JIT
    "JITError", "JITWarning",
    "JITCompilationError", "JITCompilationWarning", "CompilationWarning",
    "FormulaParsingError", "FormulaParsingWarning",
    # Config
    "ConfigError", "ConfigWarning",
    "NetworkStructureError", "NetworkStructureWarning",
    "LayerConfigurationError", "LayerConfigurationWarning",
    "PathError", "PathWarning",
    # Wrapper
    "WrapperError", "WrapperWarning",
    "TrainingError", "TrainingWarning",
    "LoadingError", "LoadingWarning",
    "SavingError", "SavingWarning",
    # General
    "RuntimeStateError", "RuntimeStateWarning",
    "ShapeMismatchError", "ShapeMismatchWarning", "ShapeWarning",
    "DataTypeError", "DataTypeWarning",
    "PerformanceWarning",
]

class HeteroSymNNError(Exception):
    """
    Base class for all HeteroSymNN exceptions.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class HeteroSymNNWarnings(HeteroSymNNError, UserWarning):
    """
    Base class for all HeteroSymNN warnings.
    """
    pass

class BackendError(HeteroSymNNError):
    """
    Base for backend-related errors.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class BackendWarning(BackendError, HeteroSymNNWarnings):
    """
    Base for backend-related warnings.
    """
    pass

class JITError(HeteroSymNNError):
    """
    Base for JIT compilation errors.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class JITWarning(JITError, HeteroSymNNWarnings):
    """
    Base for JIT compilation warnings.
    """
    pass

class ConfigError(HeteroSymNNError):
    """
    Base for configuration errors.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class ConfigWarning(ConfigError, HeteroSymNNWarnings):
    """
    Base for configuration warnings.
    """
    pass

class WrapperError(HeteroSymNNError):
    """
    Base for wrapper-related errors.
    """
    pass

class WrapperWarning(WrapperError, HeteroSymNNWarnings):
    """
    Base for wrapper-related warnings.
    """
    pass

class RuntimeStateError(HeteroSymNNError):
    """
    Exception raised when a method is called in an invalid execution state or order 
    (e.g., calling a backward pass before a forward pass).
    """
    pass

class RuntimeStateWarning(RuntimeStateError, HeteroSymNNWarnings):
    """
    Warning corresponding to RuntimeStateError.
    """
    pass

class MethodMigrationError(BackendError):
    """
    Exception raised when a computational method migration fails.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class MethodMigrationWarning(MethodMigrationError, BackendWarning):
    """
    Warning corresponding to MethodMigrationError.
    """
    pass

class ShapeMismatchError(HeteroSymNNError):
    """
    Exception raised when there is a mismatch in the shapes of the receiving data.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class ShapeMismatchWarning(ShapeMismatchError, HeteroSymNNWarnings):
    """
    Warning corresponding to ShapeMismatchError.
    """
    pass

# Keep original name ShapeWarning for backwards compatibility
class ShapeWarning(ShapeMismatchWarning):
    """
    Warning raised when there is a mismatch in the shapes of the receiving data.
    """
    pass

class BackendNotAvailableError(BackendError):
    """
    Exception raised when a backend is not available.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class BackendNotAvailableWarning(BackendNotAvailableError, BackendWarning):
    """
    Warning corresponding to BackendNotAvailableError.
    """
    pass

class InvalidDeviceIDError(BackendError):
    """
    Exception raised when an invalid device ID is provided.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class InvalidDeviceIDWarning(InvalidDeviceIDError, BackendWarning):
    """
    Warning corresponding to InvalidDeviceIDError.
    """
    pass

class ResourceAllocationError(BackendError):
    """
    Exception raised when the hardware (specifically GPU VRAM) runs out of memory. 
    Usually indicates the batch size is too large or the network is too deep for the current device.
    """
    pass

class ResourceAllocationWarning(ResourceAllocationError, BackendWarning):
    """
    Warning corresponding to ResourceAllocationError.
    """
    pass

class BackendDataTypeError(BackendError):
    """
    Exception raised when a tensor's data type (e.g., float64, int32) is incompatible 
    with the current backend's expected precision (e.g., float32), 
    preventing C++ JIT or CUDA kernel execution.
    """
    pass

class BackendDataTypeWarning(BackendDataTypeError, BackendWarning):
    """
    Warning corresponding to BackendDataTypeError.
    """
    pass

class JITCompilationError(JITError):
    """
    Exception raised when JIT compilation fails.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class JITCompilationWarning(JITCompilationError, JITWarning):
    """
    Warning corresponding to JITCompilationError.
    """
    pass

# Keep original name CompilationWarning for backwards compatibility
class CompilationWarning(JITCompilationWarning):
    """
    Warning raised when there was a problem with the creation of the kernels.
    """
    pass

class FormulaParsingError(JITError):
    """
    Exception raised when a formula parsing fails.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class FormulaParsingWarning(FormulaParsingError, JITWarning):
    """
    Warning corresponding to FormulaParsingError.
    """
    pass

class NetworkStructureError(ConfigError):
    """
    Exception raised when there is an issue with the network structure.
    Invalid number of layer or activation mismatches.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class NetworkStructureWarning(NetworkStructureError, ConfigWarning):
    """
    Warning corresponding to NetworkStructureError.
    """
    pass

class LayerConfigurationError(ConfigError):
    """
    Exception raised when there is an issue with the layer configuration.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class LayerConfigurationWarning(LayerConfigurationError, ConfigWarning):
    """
    Warning corresponding to LayerConfigurationError.
    """
    pass

class PathError(ConfigError):
    """
    Exception raised when there is an issue with the path.
    """
    pass

class PathWarning(PathError, ConfigWarning):
    """
    Warning where there is a problem with a path.
    """
    pass

class TrainingError(WrapperError):
    """
    Exception raised when an error occurs during the training process within a wrapper.
    """
    pass

class TrainingWarning(TrainingError, WrapperWarning):
    """
    Warning corresponding to TrainingError.
    """
    pass

class LoadingError(WrapperError):
    """
    Exception raised when an error occurs during the loading process of models.
    """
    pass

class LoadingWarning(LoadingError, WrapperWarning):
    """
    Warning raised when an issue occurs during the loading process that is not fatal.
    """
    pass

class SavingError(WrapperError):
    """
    Exception raised when an error occurs during the saving process of models.
    """
    pass

class SavingWarning(SavingError, WrapperWarning):
    """
    Warning corresponding to SavingError.
    """
    pass

class DataTypeError(HeteroSymNNError):
    """
    HeteroSymNN exception for type errors.
    """
    pass

class DataTypeWarning(DataTypeError, HeteroSymNNWarnings):
    """
    Warning corresponding to DataTypeError.
    """
    pass

class PerformanceWarning(HeteroSymNNWarnings):
    """
    Warning raised when a configuration or operation might lead to suboptimal performance.
    """
    pass

class HardwareWarning(BackendWarning):
    """
    Warning raised when there is a potential issue or limitation with the hardware 
    (e.g., falling back to CPU when GPU is requested).
    """
    pass
