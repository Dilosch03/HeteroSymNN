import inspect
import importlib
import pkgutil


from ..Core import losses, optimizers,initializers,layers, Nets
from . import utilities
from ..exceptions import DataTypeError

class _Registry:
    """
    Class to manage the dictionaries for the loading of saved models.


    """
    def __init__(self):
        self._loss_fn_map:dict[str,losses.Loss] = self._build_dynamic_map(losses, losses.Loss)
        self._optimizers_map:dict[str,optimizers.Optimizer] = self._build_dynamic_map(optimizers, optimizers.Optimizer)
        self._initializers_map:dict[str, initializers.Initializer] = self._build_dynamic_map(initializers, initializers.Initializer)
        self._layers_map:dict[str, layers.BaseLayer] = self._build_dynamic_map(layers, layers.BaseLayer)
        self._net_map:dict[str,Nets.BaseNetwork] = self._scan_package_deep(Nets,Nets.BaseNetwork)
        self._transformers_map:dict[str,utilities.DataTransformer] = self._build_dynamic_map(utilities, utilities.DataTransformer)

    def _build_dynamic_map(self,module, base_class)->dict[str,type]:
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
    
    def _scan_package_deep(self,package, base_class):
        """
        Recursively scans a package folder and all its .py files for subclasses.
        
        """
        found_classes = {}
        
        # 1. Walk through every module (.py file) in the directory
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            # 2. Dynamically import the file (e.g., 'HeteroSymNN.Core.Nets.dense')
            full_module_name = f"{package.__name__}.{module_name}"
            module = importlib.import_module(full_module_name)
            
            # 3. Inspect the newly loaded file
            for name, member in inspect.getmembers(module):
                if inspect.isclass(member) and issubclass(member, base_class) and member is not base_class:
                    found_classes[name] = member
                    
        return found_classes
    
    @property
    def loss_fn_map(self):
        return self._loss_fn_map
    
    @property
    def optimiers_map(self):
        return self._optimizers_map
    
    @property
    def initializers_map(self):
        return self._initializers_map
    
    @property
    def layers_map(self):
        return self._layers_map
    
    @property
    def net_map(self):
        return self._net_map

    @property
    def data_transformers_map(self):
        return self._transformers_map
    
    def add_initializer(self,custom_initializer:type[initializers.Initializer]):
        if not(issubclass(custom_initializer,initializers.Initializer)):
            raise DataTypeError("The class that was given is not a child of the base initializer class (initializers.Initializer).")
        self._initializers_map[custom_initializer.__name__] = custom_initializer
    
    def add_loss_func(self,custom_loss:type[losses.Loss]):
        if not(issubclass(custom_loss,losses.Loss)):
            raise DataTypeError("The class that was given is not a child of the base loss function class (losses.Loss).")
        self._loss_fn_map[custom_loss.__name__] = custom_loss
    
    def add_optimizer(self,custom_optimizer:type[optimizers.Optimizer]):
        if not(issubclass(custom_optimizer,optimizers.Optimizer)):
            raise DataTypeError("The class that was given is not a child of the base optimizer class (optimizers.Optimizer).")
        self._optimizers_map[custom_optimizer.__name__] = custom_optimizer

    def add_layer(self,custom_layer:type[layers.BaseLayer]):
        if not(issubclass(custom_layer,layers.BaseLayer)):
            raise DataTypeError("The class that was given is not a child of the base layer class (layers.BaseLayer).")
        self._layers_map[custom_layer.__name__] = custom_layer

    def add_net(self,custom_net:type[Nets.BaseNetwork]):
        if not(issubclass(custom_net,Nets.BaseNetwork)):
            raise DataTypeError("The class that was given is not a child of the base network class (Nets.BaseNetwork).")
        self._net_map[custom_net.__name__] = custom_net
    
    def add_data_transformer(self,custom_data_transformer:type[utilities.DataTransformer]):
        if not(issubclass(custom_data_transformer,utilities.DataTransformer)):
            raise DataTypeError("The class that was given is not a child of the base data transformer class (utilities.DataTransformer).")
        self._net_map[custom_data_transformer.__name__] = custom_data_transformer

registry = _Registry()
