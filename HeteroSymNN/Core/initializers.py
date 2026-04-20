import numpy as np
from typing import Any

from ..config import settings

class Initializer:
    """
    Base class for all initializers.
    
    Should act as a shape-filling utility for Layers. The Layer dictates the shape.
    """
    def __init__(self):
        pass
        
    def generate_from_distribution(self, shape:list[int],fan_in: int, fan_out: int) -> np.ndarray:
        """
        Fills a tensor of `shape` with initial values based on statistical distributions.

        This method must be implemented by subclasses to define the specific initialization logic.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.
        
        fan_in : int
            Number of input units.

        fan_out : int
            Number of output units.


        Returns
        -------
        `np.ndarray`
            numpy array populated with values drawn from the statistical distribution.
        """
        raise NotImplementedError

    def generate_constant(self, shape:list[int], value:float=0.0) -> np.ndarray:
        """
        Fills a tensor of `shape` with a constant value (zeros by default).

        This method must be implemented by subclasses to define the specific initialization logic.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.
        
        value : float, optional
            The constant value to fill the array with. Default is 0.0.


        Returns
        -------
        `np.ndarray`
            numpy array populated with the specified constant value.
        """
        raise NotImplementedError

    def generate_binary_mask(self, shape:list[int])-> np.ndarray:
        """
        Generates a binary dropout/sparsity mask of `shape` based on density.

        This method must be implemented by subclasses to define the specific initialization logic.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.


        Returns
        -------
        `np.ndarray`
            numpy array populated with binary values (1.0 or 0.0) representing the sparsity mask.
        """
        raise NotImplementedError

    def get_config(self)->dict[str,Any]:
        """
        Returns the configuration of the initializer.

        This method should be implemented by subclasses to return a dictionary containing 
        the configuration parameters necessary to reconstruct the initializer instance.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the configuration parameters.
        """
        return {"class_name": self.__class__.__name__}

class BaseInitializer(Initializer):
    """
    Base class for initializers that provides generic definitions of how the `generate_constant` and `generate_binary_mask` for general methods.
    
    Acts as a shape-filling utility for Layers. The Layer dictates the shape.

    Parameters
    ----------
    connection_density: float, optional
        defines how sparced the binary mask is going to be. Defaults to 1.0.
    """
    def __init__(self,connection_density:float=None):
        self._seted_connection_density = connection_density
        if (connection_density is None):
            connection_density = 1.0
        self.connection_density = np.clip(connection_density, 0.0, 1.0,dtype=settings.default_dtype)
    
    def generate_constant(self, shape:list[int], value:float = 0.0) -> np.ndarray:
        """
        Fills a tensor of `shape` with a constant value (zeros by default).

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.
        
        value : float, optional
            The constant value to fill the array with. Default is 0.0.


        Returns
        -------
        `np.ndarray`
            numpy array populated with the specified constant value.
        """
        return np.full(shape, value, dtype=settings.default_dtype)

    def generate_binary_mask(self, shape:list[int])-> np.ndarray:
        """
        Generates a binary dropout/sparsity mask of `shape` based on density.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.


        Returns
        -------
        `np.ndarray`
            numpy array populated with binary values (1.0 or 0.0) representing the sparsity mask.
        """
        if (self.connection_density) >= 1.0:
            return np.ones(shape, dtype=settings.default_dtype)
        return (np.random.rand(*shape) < self.connection_density).astype(settings.default_dtype)

    def get_config(self)->dict[str,Any]:
        """
        Returns the configuration of the initializer.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the configuration parameters.

            *"class_name": for saving of the initializer.

            *"connection_density": how dense the binary mask creates.
        """
        return {"class_name": self.__class__.__name__, "connection_density": self.connection_density}

class RandomNormal(BaseInitializer):
    """
    Initializer that generates tensors with a normal distribution.
    
    Acts as a shape-filling utility for Layers. The Layer dictates the shape.
    
    Recommended for generic, custom, or un-normalized layers where bounded noise is needed.

    Parameters
    ----------
    mean : float, optional
        Mean of the random values to generate. Defaults to 0.0.
    stddev : float, optional
        Standard deviation of the random values to generate. Defaults to 0.05.
    """
    def __init__(self, mean:float=0.0, stddev:float=0.05, connection_density:float=None):
        super().__init__(connection_density)
        self.mean = np.array(mean,dtype=settings.default_dtype)
        self.stddev = np.array(stddev,dtype=settings.default_dtype)
    
    def generate_from_distribution(self, shape:list[int],fan_in: int, fan_out: int) -> np.ndarray:
        """
        Fills a tensor of `shape` with initial values using a normal distribution.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.
        
        fan_in : int
            Number of input units.

        fan_out : int
            Number of output units.


        Returns
        -------
        `np.ndarray`
            numpy array populated with values drawn from the statistical distribution.
        """
        return np.random.normal(self.mean, self.stddev, shape).astype(settings.default_dtype)
    
    def get_config(self)->dict[str,Any]:
        """
        Returns the configuration of the initializer.

        This method should be implemented by subclasses to return a dictionary containing 
        the configuration parameters necessary to reconstruct the initializer instance.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the configuration parameters.

            *"class_name": for saving of the initializer.

            *"connection_density": how dense the binary mask creates.

            *"stddev": standard deviation of the random values to generate.

            *"mean": mean of the random values to generate.

        """
        params = {"stddev": self.stddev, "mean": self.mean}
        params.update(super().get_config())
        return params


class RandomUniform(BaseInitializer):
    """
    Initializer that generates tensors with a uniform distribution.
    
    Acts as a shape-filling utility for Layers. The Layer dictates the shape.
    
    Recommended for generic, custom, or un-normalized layers where strictly bounded noise is needed.

    Parameters
    ----------
    min_val : float, optional
        Lower bound of the range of random values to generate. Defaults to -0.05.
    max_val : float, optional
        Upper bound of the range of random values to generate. Defaults to 0.05.
    """
    def __init__(self, min_val:float=-0.05, max_val:float=0.05,connection_density:float=None):
        super().__init__(connection_density)
        self.min_val = np.array(min_val,dtype=settings.default_dtype)
        self.max_val = np.array(max_val,dtype=settings.default_dtype)
    
    def generate_from_distribution(self, shape:list[int],fan_in: int, fan_out: int) -> np.ndarray:
        """
        Fills a tensor of `shape` with initial values using a uniform distribution.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.
        
        fan_in : int
            Number of input units.

        fan_out : int
            Number of output units.


        Returns
        -------
        `np.ndarray`
            numpy array populated with values drawn from the statistical distribution.
        """
        return np.random.uniform(self.min_val, self.max_val, shape).astype(settings.default_dtype)

    def get_config(self)->dict[str,Any]:
        """
        Returns the configuration of the initializer.

        This method should be implemented by subclasses to return a dictionary containing 
        the configuration parameters necessary to reconstruct the initializer instance.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the configuration parameters.

            *"class_name": for saving of the initializer.

            *"connection_density": how dense the binary mask creates.

            *"min_val": lower bound of the range of random values to generate.

            *"max_val": upper bound of the range of random values to generate.

        """
        params = {"min_val": self.min_val, "max_val": self.max_val}
        params.update(super().get_config())
        return params

class XavierUniform(BaseInitializer):
    """
    Xavier (Glorot) uniform initializer.
    
    Acts as a shape-filling utility for Layers. The Layer dictates the shape.
    
    Recommended if using Sigmoid, Tanh, or Softmax activation functions.
    
    Draws samples from a uniform distribution within [-limit, limit] where `limit` is `sqrt(6 / (fan_in + fan_out))`.
    """
    def generate_from_distribution(self, shape:list[int],fan_in: int, fan_out: int) -> np.ndarray:
        """
        Fills a tensor of `shape` with initial values using a Unifrom Xavier distribution.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.
        
        fan_in : int
            Number of input units.

        fan_out : int
            Number of output units.


        Returns
        -------
        `np.ndarray`
            numpy array populated with values drawn from the statistical distribution.
        """
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape).astype(settings.default_dtype)

class XavierNormal(BaseInitializer):
    """
    Xavier (Glorot) normal initializer.
    
    Acts as a shape-filling utility for Layers. The Layer dictates the shape.
    
    Recommended if using Sigmoid, Tanh, or Softmax activation functions.
    
    Draws samples from a normal distribution centered on 0 with `stddev = sqrt(2 / (fan_in + fan_out))`.
    """
    def generate_from_distribution(self, shape:list[int],fan_in: int, fan_out: int) -> np.ndarray:
        """
        Fills a tensor of `shape` with initial values using a Normal Xavier distribution.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.
        
        fan_in : int
            Number of input units.

        fan_out : int
            Number of output units.


        Returns
        -------
        `np.ndarray`
            numpy array populated with values drawn from the statistical distribution.
        """
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, stddev, shape).astype(settings.default_dtype)

class HeUniform(BaseInitializer):
    """
    He uniform variance scaling initializer.
    
    Acts as a shape-filling utility for Layers. The Layer dictates the shape.
    
    Recommended if using ReLU, LeakyReLU, and ELU activation functions.
    
    Draws samples from a uniform distribution within [-limit, limit] where `limit` is `sqrt(6 / fan_in)`.
    """
    def generate_from_distribution(self, shape:list[int],fan_in: int, fan_out: int) -> np.ndarray:
        """
        Fills a tensor of `shape` with initial values using a Unifrom He distribution.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.
        
        fan_in : int
            Number of input units.

        fan_out : int
            Number of output units.


        Returns
        -------
        `np.ndarray`
            numpy array populated with values drawn from the statistical distribution.
        """
        limit = np.sqrt(6 / fan_in)
        return np.random.uniform(-limit, limit, shape).astype(settings.default_dtype)

class HeNormal(BaseInitializer):
    """
    He normal initializer.
    
    Acts as a shape-filling utility for Layers. The Layer dictates the shape.

    Recommended if using ReLU, LeakyReLU, and ELU activation functions.
    
    Draws samples from a normal distribution centered on 0 with `stddev = sqrt(2 / fan_in)`.
    """
    def generate_from_distribution(self, shape:list[int],fan_in: int, fan_out: int) -> np.ndarray:
        """
        Fills a tensor of `shape` with initial values using a Normal He distribution.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.
        
        fan_in : int
            Number of input units.

        fan_out : int
            Number of output units.


        Returns
        -------
        `np.ndarray`
            numpy array populated with values drawn from the statistical distribution.
        """
        stddev = np.sqrt(2 / fan_in)
        return np.random.normal(0, stddev, shape).astype(settings.default_dtype)

class LecunNormal(BaseInitializer):
    """
    LeCun normal initializer.
    
    Acts as a shape-filling utility for Layers. The Layer dictates the shape.

    Recommended if using SELU activations or for heterogeneous/mixed activation layers.
    
    Draws samples from a normal distribution centered on 0 with `stddev = sqrt(1 / fan_in)`.
    """
    def generate_from_distribution(self, shape:list[int],fan_in: int, fan_out: int) -> np.ndarray:
        """
        Fills a tensor of `shape` with initial values using a Normal LeCun distribution.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.
        
        fan_in : int
            Number of input units.

        fan_out : int
            Number of output units.


        Returns
        -------
        `np.ndarray`
            numpy array populated with values drawn from the statistical distribution.
        """
        stddev = np.sqrt(1 / fan_in)
        return np.random.normal(0, stddev, shape).astype(settings.default_dtype)

class Orthogonal(BaseInitializer):
    """
    Initializer that generates an orthogonal matrix.
    
    Acts as a shape-filling utility for Layers. The Layer dictates the shape.

    Recommended for Recurrent Neural Networks (RNNs) or using heterogeneous layers.
    
    Parameters
    ----------
    gain : float, optional
        Multiplicative factor to apply to the orthogonal matrix. Defaults to 1.0.
    """
    def __init__(self, gain:float=1.0,connection_density:float=None):
        super().__init__(connection_density)
        self.gain = np.array(gain,dtype=settings.default_dtype)

    def generate_from_distribution(self, shape:list[int],fan_in: int, fan_out: int) -> np.ndarray:
        """
        Fills a tensor of `shape` with an orthogonal matrix scaled by a gain factor.

        Parameters
        ----------
        shape : list[int]
            Shape of the numpy array that will be generated.
        
        fan_in : int
            Number of input units.

        fan_out : int
            Number of output units.


        Returns
        -------
        `np.ndarray`
            numpy array populated with values forming an orthogonal matrix.
        """
        flat_shape = (shape[0], int(np.prod(shape[1:]))) if len(shape) > 1 else (shape[0], 1)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return (self.gain * q).reshape(shape).astype(settings.default_dtype)

    def get_config(self)->dict[str,Any]:
        """
        Returns the configuration of the initializer.

        This method should be implemented by subclasses to return a dictionary containing 
        the configuration parameters necessary to reconstruct the initializer instance.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the configuration parameters.

            *"class_name": for saving of the initializer.

            *"connection_density": how dense the binary mask creates.

            *"gain": Multiplicative factor to applied to the orthogonal matrix.

        """
        params = {"gain": self.gain}
        params.update(super().get_config())
        return params