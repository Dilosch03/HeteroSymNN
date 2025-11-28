import numpy as np

from ..types import LayerValues

class Initializer:
    def __init__(self):
        pass
        
    def generate(self, fan_in: int, fan_out: int) -> LayerValues:
        raise NotImplementedError

    def get_config(self):
        return {"class_name": self.__class__.__name__}

class BaseInitializer(Initializer):
    def _create_matrices(self, fan_in: int, fan_out: int, weights: np.ndarray):
        biases = [0.0] * fan_out
        connection_mask = np.ones((fan_out, fan_in))
        return (biases, weights.tolist(), connection_mask.tolist())

class RandomNormal(BaseInitializer):
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev

    def generate(self, fan_in: int, fan_out: int):
        weights = np.random.normal(self.mean, self.stddev, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)
    
    def get_config(self):
        return {"class_name": self.__class__.__name__, "mean": self.mean, "stddev": self.stddev}


class RandomUniform(BaseInitializer):
    def __init__(self, min_val=-0.05, max_val=0.05):
        self.min_val = min_val
        self.max_val = max_val

    def generate(self, fan_in: int, fan_out: int):
        weights = np.random.uniform(self.min_val, self.max_val, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

    def get_config(self):
        return {"class_name": self.__class__.__name__, "min_val": self.min_val, "max_val": self.max_val}

# Los siguientes no tienen parámetros configurables en __init__, usan el get_config base
class XavierUniform(BaseInitializer):
    def generate(self, fan_in: int, fan_out: int):
        limit = np.sqrt(6 / (fan_in + fan_out))
        weights = np.random.uniform(-limit, limit, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

class XavierNormal(BaseInitializer):
    def generate(self, fan_in: int, fan_out: int):
        stddev = np.sqrt(2 / (fan_in + fan_out))
        weights = np.random.normal(0, stddev, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

class HeUniform(BaseInitializer):
    def generate(self, fan_in: int, fan_out: int):
        limit = np.sqrt(6 / fan_in)
        weights = np.random.uniform(-limit, limit, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

class HeNormal(BaseInitializer):
    def generate(self, fan_in: int, fan_out: int):
        stddev = np.sqrt(2 / fan_in)
        weights = np.random.normal(0, stddev, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

class LecunNormal(BaseInitializer):
    def generate(self, fan_in: int, fan_out: int):
        stddev = np.sqrt(1 / fan_in)
        weights = np.random.normal(0, stddev, (fan_out, fan_in))
        return self._create_matrices(fan_in, fan_out, weights)

class Orthogonal(BaseInitializer):
    def __init__(self, gain=1.0):
        self.gain = gain

    def generate(self, fan_in: int, fan_out: int):
        flat_shape = (fan_out, fan_in)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        weights = self.gain * q.reshape(flat_shape)
        return self._create_matrices(fan_in, fan_out, weights)

    def get_config(self):
        return {"class_name": self.__class__.__name__, "gain": self.gain}