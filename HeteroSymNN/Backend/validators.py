from . import hardware as HW
from ..exceptions import InvalidDeviceIDError

__all__ = []

def _validate_gpu_id(gpu_id: int) -> None:
    """
    Validates that a GPU device ID is within the valid range.

    Parameters
    ----------
    gpu_id : int
        GPU device ID to validate.

    Raises
    ------
    :exc:`~HeteroSymNN.exceptions.InvalidDeviceIDError`
        If the GPU ID is negative or greater than or equal to the number of available GPUs.
    """
    if (gpu_id < 0 or gpu_id >= HW.NUM_GPUS):
        raise InvalidDeviceIDError(
            f"GPU ID {gpu_id} is out of range. "
            f"Valid IDs are 0..{HW.NUM_GPUS - 1} ({HW.NUM_GPUS} GPU(s) detected)."
        )
