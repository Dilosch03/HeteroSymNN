import numpy as np
from typing import Any

from ..exceptions import RuntimeStateError
from ..error_handlers import clean_traceback

__all__ = ["DataTransformer", "MinMaxScaler", "StandardScaler"]

class DataTransformer:
    """
    Base class for data transformations.
    """
    def __init__(self):
        self._fitted = False
    
    @property
    def is_fitted(self)->bool:
        """
        Property to get if the transformer as been fitted.

        Returns
        -------
        bool
        """
        return self._fitted

    def fit(self,data:np.ndarray):
        """
        Method to fit the transformer using the data.

        The implementation must be done in the subclass.
        
        Parameters
        ----------
        data : np.ndarray
            Data to extract the parameters nessesary for the transformation.
        """
        self._fitted = True
        return self

    def transform(self, data:np.ndarray)->np.ndarray:
        """
        Method to transform data.
        
        Parameters
        ----------
        data : np.ndarray
            Data to transform.
        
        Returns
        -------
        np.ndarray
            transformed data.
        
        Raises
        ---------
        RuntimeStateError
            If the scaler is not fitted.
        """
        if not(self._fitted):
            raise RuntimeStateError("Trying to do data scaling before fitting the scaler.")

    def inverse_transform(self, data:np.ndarray)->np.ndarray:
        """
        Method to reverse the transformation of the data.
        
        Parameters
        ----------
        data : np.ndarray
            Data to do the inverse transformation.
        
        Returns
        -------
        np.ndarray
            Detransformed data.
        
        Raises
        ---------
        RuntimeStateError
            If the scaler is not fitted.
        """
        if not(self._fitted):
            raise RuntimeStateError("Trying to do data descaling before fitting the scaler.")
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Method to fit and transform the given data.
        
        Parameters
        ----------
        data : np.ndarray
            Data to transform.
        
        Returns
        -------
        np.ndarray
            Transformed data.
        """
        self.fit(data)
        return self.transform(data)

    
    def get_config(self)->dict[str,Any]:
        """
        Method to get the configuration of the instance.

        Needs to return the values of the internal parameters of the instance for it to be able to reconstruct itself with out needing the dataset.
        
        Returns
        -------
        dict[str,Any]
            Configuration of the transformer.
        """
        return {"fitted":self._fitted}
    
    def set_config(self,config:dict[str,Any])->None:
        """
        Method to set the configuration of the instance.

        From a dictionary with strings as keys be able to reconstruct the transformer with out needing to pass the data.

        Parameters
        ----------
        config : dict[str,Any]
            Configuration of the transformer.
        """
        self._fitted = config["fitted"]

class MinMaxScaler(DataTransformer):
    """
    Transformer class that normalizes data between 0 and 1 with a min and max value method.
    """

    def __init__(self):
        super().__init__()
        self._min = None
        self._max = None

    @property
    def min(self)->float:
        """
        Property to get the min value of the instance.

        Returns
        -------
        float
        """
        return self._min
    
    @property
    def max(self)->float:
        """
        Property to get the max value of the instance.

        Returns
        -------
        float
        """
        return self._max
    
    def fit(self,data:np.ndarray)->None:
        """
        Method to fit the transformer using the min max method.

        Parameters
        ----------
        data: np.ndarray
            Data to extract the min and max values of the dataset.
        """
        self._min = np.min(data)
        self._max = np.max(data)
        super().fit(data)

    @clean_traceback
    def transform(self, data:np.ndarray)->np.ndarray:
        super().transform(data)
        range_val = self._max - self._min
        if np.any(range_val == 0):
            with np.errstate(invalid='ignore', divide='ignore'):
                return np.where(range_val == 0, 0.0, (data - self._min) / range_val)
        return (data - self._min) / range_val

    @clean_traceback
    def inverse_transform(self, data:np.ndarray)->np.ndarray:
        super().inverse_transform(data)
        return data * (self._max - self._min) + self._min
    
    def get_config(self)->dict[str,float]:
        """
        Method to get the configuration of the instance.
        
        Returns
        -------
        dict[str,float]
            Min and max values of the instance.
        """
        config = super().get_config()
        config.update({"min":self._min, "max":self._max})
        return config
    
    def set_config(self,config:dict[str,float])->None:
        """
        Method to set the configuration of the instance.
        
        Parameters
        ----------
        config : dict[str,float]
            Min and max values of the instance.
        """
        super().set_config(config)
        self._min = config["min"]
        self._max = config["max"]

class StandardScaler(DataTransformer):
    """
    Transformer class that standardizes data by removing the mean and scaling to unit variance.

    Applies the z-score normalization formula per feature (column):

    .. math::
        z = \\frac{x - \\mu}{\\sigma}

    where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation
    computed from the training data.

    This is equivalent to ``sklearn.preprocessing.StandardScaler``.
    """

    def __init__(self):
        super().__init__()
        self._mean = None
        self._std = None

    @property
    def mean(self)->np.ndarray:
        """
        Property to get the per-feature mean computed during fitting.

        Returns
        -------
        np.ndarray or None
        """
        return self._mean
    
    @property
    def std(self)->np.ndarray:
        """
        Property to get the per-feature standard deviation computed during fitting.

        Returns
        -------
        np.ndarray or None
        """
        return self._std
    
    def fit(self, data:np.ndarray)->None:
        """
        Computes the mean and standard deviation per feature from the training data.

        Parameters
        ----------
        data : np.ndarray
            Data to extract the mean and standard deviation of. 
            Shape should be ``(n_samples,)`` or ``(n_samples, n_features)``.
        """
        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)
        super().fit(data)

    @clean_traceback
    def transform(self, data:np.ndarray)->np.ndarray:
        """
        Standardizes data using the fitted mean and standard deviation.

        Parameters
        ----------
        data : np.ndarray
            Data to transform.
        
        Returns
        -------
        np.ndarray
            Standardized data with zero mean and unit variance.
        """
        super().transform(data)
        if np.any(self._std == 0):
            with np.errstate(invalid='ignore', divide='ignore'):
                return np.where(self._std == 0, 0.0, (data - self._mean) / self._std)
        return (data - self._mean) / self._std

    @clean_traceback
    def inverse_transform(self, data:np.ndarray)->np.ndarray:
        """
        Reverses the standardization, recovering the original scale.

        Parameters
        ----------
        data : np.ndarray
            Standardized data to inverse-transform.
        
        Returns
        -------
        np.ndarray
            Data in the original scale.
        """
        super().inverse_transform(data)
        return data * self._std + self._mean
    
    def get_config(self)->dict[str, Any]:
        """
        Method to get the configuration of the instance.
        
        Returns
        -------
        dict[str, Any]
            Mean and standard deviation values of the instance.
        """
        config = super().get_config()
        config.update({
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
        })
        return config
    
    def set_config(self, config:dict[str, Any])->None:
        """
        Method to set the configuration of the instance.
        
        Parameters
        ----------
        config : dict[str, Any]
            Mean and standard deviation values of the instance.
        """
        super().set_config(config)
        self._mean = np.array(config["mean"]) if config["mean"] is not None else None
        self._std = np.array(config["std"]) if config["std"] is not None else None