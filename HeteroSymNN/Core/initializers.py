import numpy as np

from ..types import LayerValues

class Initializer:
    """
    Base class for all initializers.
    """
    def __init__(self):
        """
        In the case that the initializers need extra parameters overload the __init__ method.
        """
        pass
        
    def generate(self, fan_in: int, fan_out: int) -> LayerValues:
        """
        Generates the initial values for a layer.

        This method must be implemented by subclasses to define the specific initialization logic.

        Parameters
        ----------
        fan_in : int
            Number of input units.
        fan_out : int
            Number of output units.

        Returns
        -------
        :obj:`~HeteroSymNN.types.LayerValues`
            Tuple containing biases, weights, and connection mask.
        """
        raise NotImplementedError

    def get_config(self):
        """
        Returns the configuration of the initializer.

        This method should be implemented by subclasses to return a dictionary containing 
        the configuration parameters necessary to reconstruct the initializer instance.

        Returns
        -------
        dict
            Dictionary containing the configuration parameters.
        """
        return {"class_name": self.__class__.__name__}

class BaseInitializer(Initializer):
    """
    Base class for initializers that provides a helper method to create the return structure.
    """
    def _create_matrices(self, fan_in: int, fan_out: int, weights: np.ndarray):
        """
        Internal helper to format the generated weights into the expected LayerValues structure.
        """
        biases = [0.0] * fan_out
        connection_mask = np.ones((fan_out, fan_in))
        return (biases, weights.tolist(), connection_mask.tolist())

class RandomNormal(BaseInitializer):
    """
    Initializer that generates tensors with a normal distribution.

    Parameters
    ----------
    mean : float, optional
        Mean of the random values to generate. Defaults to 0.0.
    stddev : float, optional
        Standard deviation of the random values to generate. Defaults to 0.05.
    """
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev

    def generate(self, fan_in: int, fan_out: int):
        """
        Generates the initial values.

        Parameters
        ----------
        fan_in : int
            Number of input units.
        fan_out : int
            Number of output units.

        Returns
        -------
        :obj:`~HeteroSymNN.types.LayerValues`
            Tuple containing biases, weights, and connection mask.
        """
        weights = np.random.normal(self.mean, self.stddev, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)
    
    def get_config(self):
        return {"class_name": self.__class__.__name__, "mean": self.mean, "stddev": self.stddev}


class RandomUniform(BaseInitializer):
    """
    Initializer that generates tensors with a uniform distribution.

    Parameters
    ----------
    min_val : float, optional
        Lower bound of the range of random values to generate. Defaults to -0.05.
    max_val : float, optional
        Upper bound of the range of random values to generate. Defaults to 0.05.
    """
    def __init__(self, min_val=-0.05, max_val=0.05):
        self.min_val = min_val
        self.max_val = max_val

    def generate(self, fan_in: int, fan_out: int):
        """
        Generates the initial values.

        Parameters
        ----------
        fan_in : int
            Number of input units.
        fan_out : int
            Number of output units.

        Returns
        -------
        :obj:`~HeteroSymNN.types.LayerValues`
            Tuple containing biases, weights, and connection mask.
        """
        weights = np.random.uniform(self.min_val, self.max_val, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

    def get_config(self):
        return {"class_name": self.__class__.__name__, "min_val": self.min_val, "max_val": self.max_val}

class XavierUniform(BaseInitializer):
    """
    Xavier (Glorot) uniform initializer.
    
    Draws samples from a uniform distribution within [-limit, limit] where `limit` is `sqrt(6 / (fan_in + fan_out))`.
    """
    def generate(self, fan_in: int, fan_out: int):
        """
        Generates the initial values.

        Parameters
        ----------
        fan_in : int
            Number of input units.
        fan_out : int
            Number of output units.

        Returns
        -------
        :obj:`~HeteroSymNN.types.LayerValues`
            Tuple containing biases, weights, and connection mask.
        """
        limit = np.sqrt(6 / (fan_in + fan_out))
        weights = np.random.uniform(-limit, limit, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

class XavierNormal(BaseInitializer):
    """
    Xavier (Glorot) normal initializer.
    
    Draws samples from a normal distribution centered on 0 with `stddev = sqrt(2 / (fan_in + fan_out))`.
    """
    def generate(self, fan_in: int, fan_out: int):
        """
        Generates the initial values.

        Parameters
        ----------
        fan_in : int
            Number of input units.
        fan_out : int
            Number of output units.

        Returns
        -------
        :obj:`~HeteroSymNN.types.LayerValues`
            Tuple containing biases, weights, and connection mask.
        """
        stddev = np.sqrt(2 / (fan_in + fan_out))
        weights = np.random.normal(0, stddev, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

class HeUniform(BaseInitializer):
    """
    He uniform variance scaling initializer.
    
    Draws samples from a uniform distribution within [-limit, limit] where `limit` is `sqrt(6 / fan_in)`.
    """
    def generate(self, fan_in: int, fan_out: int):
        """
        Generates the initial values.

        Parameters
        ----------
        fan_in : int
            Number of input units.
        fan_out : int
            Number of output units.

        Returns
        -------
        :obj:`~HeteroSymNN.types.LayerValues`
            Tuple containing biases, weights, and connection mask.
        """
        limit = np.sqrt(6 / fan_in)
        weights = np.random.uniform(-limit, limit, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

class HeNormal(BaseInitializer):
    """
    He normal initializer.
    
    Draws samples from a normal distribution centered on 0 with `stddev = sqrt(2 / fan_in)`.
    """
    def generate(self, fan_in: int, fan_out: int):
        """
        Generates the initial values.

        Parameters
        ----------
        fan_in : int
            Number of input units.
        fan_out : int
            Number of output units.

        Returns
        -------
        :obj:`~HeteroSymNN.types.LayerValues`
            Tuple containing biases, weights, and connection mask.
        """
        stddev = np.sqrt(2 / fan_in)
        weights = np.random.normal(0, stddev, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

class LecunNormal(BaseInitializer):
    """
    LeCun normal initializer.
    
    Draws samples from a normal distribution centered on 0 with `stddev = sqrt(1 / fan_in)`.
    """
    def generate(self, fan_in: int, fan_out: int):
        """
        Generates the initial values.

        Parameters
        ----------
        fan_in : int
            Number of input units.
        fan_out : int
            Number of output units.

        Returns
        -------
        :obj:`~HeteroSymNN.types.LayerValues`
            Tuple containing biases, weights, and connection mask.
        """
        stddev = np.sqrt(1 / fan_in)
        weights = np.random.normal(0, stddev, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

class Orthogonal(BaseInitializer):
    """
    Initializer that generates an orthogonal matrix.
    
    Parameters
    ----------
    gain : float, optional
        Multiplicative factor to apply to the orthogonal matrix. Defaults to 1.0.
    """
    def __init__(self, gain=1.0):
        self.gain = gain

    def generate(self, fan_in: int, fan_out: int):
        """
        Generates the initial values.

        Parameters
        ----------
        fan_in : int
            Number of input units.
        fan_out : int
            Number of output units.

        Returns
        -------
        :obj:`~HeteroSymNN.types.LayerValues`
            Tuple containing biases, weights, and connection mask.
        """
        flat_shape = (fan_out, fan_in)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        weights = self.gain * q.reshape(flat_shape)
        return self._create_matrices(fan_in, fan_out, weights)

    def get_config(self):
        return {"class_name": self.__class__.__name__, "gain": self.gain}