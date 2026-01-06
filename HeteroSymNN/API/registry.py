import inspect
from ..Core import losses, optimizers,initializers

def _build_dynamic_map(module, base_class):
    """
    Internal helper to dynamically discover subclasses of a base class within a module.
    Used to populate maps for Losses, Optimizers, and Initializers.
    """
    new_map = {}
    for name, member in inspect.getmembers(module):
        if inspect.isclass(member) and \
           issubclass(member, base_class) and \
           member is not base_class:
            
            new_map[name] = member
    return new_map

LOSS_FN_MAP:dict[str,losses.Loss] = _build_dynamic_map(losses, losses.Loss)
OPTIMIZER_MAP:dict[str,optimizers.Optimizer] = _build_dynamic_map(optimizers, optimizers.Optimizer)
INITIALIZER_MAP:dict[str, initializers.Initializer] = _build_dynamic_map(initializers, initializers.Initializer)