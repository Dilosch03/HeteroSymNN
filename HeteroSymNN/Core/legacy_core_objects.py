from __future__ import annotations
import warnings


from ..types import LayerConstructionConfig
from ..exceptions import HeteroSymNNDeprecationWarning
from .layers import LinearLayer

class Layer(LinearLayer):
    """
        Base layer class mainly used for linear_net networks
        
        Parameters
        ----------
        _num_inputs : int
            Number of inputs the layer is going to receive.
        layer_configuration : :obj:`~HeteroSymNN.types.LayerConstructionConfig`
            Configuration of the layer including the node activation functions and their constants as well as the initial _weights, _biases and connection mask.
        batch_size : int, optional
            Initial batch size for the layer, by default 1.
        Gpu_id : int, optional
            Id of the GPU to use if available, by default 0.
        
        Examples
        --------
        Generaly one doesn't need to instanciate this class directly but if some want to do it, here is an example of how to do it.

        >>> from HeteroSymNN.Core.Nets.layers import Layer
        >>> from HeteroSymNN.Core.initializers import HeNormal
        >>> num_inputs = 3
        >>> num_nodes = 5
        >>> inicial_params = HeNormal().generate(3,5)
        >>> layer_config = [("relu", {}),("relu", {}),("sigmoid", {}),("relu", {}),("relu", {})]
        >>> constuctor = (layer_config,inicial_params)
        >>> layer = Layer(num_inputs,constuctor)
        
        
    """
    def __init__(self,num_inputs:int,layer_configuration:LayerConstructionConfig,batch_size:int = 1,Gpu_id:int = 0):
        warnings.warn(
            HeteroSymNNDeprecationWarning(
                deprecated_feature="Layer",
                replacement="LinearLayer",
                version_removal="0.4.0"
            ),
            stacklevel=2
        )
        super().__init__(num_inputs=num_inputs, layer_configuration=layer_configuration, batch_size=batch_size, Gpu_id=Gpu_id)
    def get_config(self)->dict[str,any]:
        temp = super().get_config()
        temp["layer_type"] = "LinearLayer"
        return temp

