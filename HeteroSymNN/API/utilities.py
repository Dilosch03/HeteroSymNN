import numpy as np
from typing import Any

from ..exceptions import RuntimeStateError

class DataTransformer:
    """
    Base class for data transformations.
    """
    def __init__(self):
        self.fitted = False

    def fit(self,data:np.ndarray):
        """
        Method to fit the transformer using the data.

        The implementation must be done in the subclass.
        
        Parameters
        ----------
        data : np.ndarray
            Data to extract the parameters nessesary for the transformation.
        """
        self.fitted = True
        return self

    def transform(self, data:np.ndarray)->np.ndarray:
        """
        Method to normalize data.
        
        Parameters
        ----------
        data : np.ndarray
            Data to normalize.
        
        Returns
        -------
        np.ndarray
            Normalized data.
        
        Exeptions
        ---------
        RuntimeStateError
            If the scaler is not fitted.
        """
        if not(self.fitted):
            raise RuntimeStateError("Trying to do data scaling before fitting the scaler.")

    def inverse_transform(self, data:np.ndarray)->np.ndarray:
        """
        Method to denormalize data.
        
        Parameters
        ----------
        data : np.ndarray
            Data to denormalize.
        
        Returns
        -------
        np.ndarray
            Denormalized data.
        
        Exeptions
        ---------
        RuntimeStateError
            If the scaler is not fitted.
        """
        if not(self.fitted):
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
        return {"fitted":self.fitted}
    
    def set_config(self,config:dict[str,Any])->None:
        """
        Method to set the configuration of the instance.

        From a dictionary with strings as keys be able to reconstruct the transformer with out needing to pass the data.

        Parameters
        ----------
        config : dict[str,Any]
            Configuration of the transformer.
        """
        self.fitted = config["fitted"]

class MinMaxScaler(DataTransformer):
    """
    Transformer class that normalizes data between 0 and 1 with a min and max value method.
    """

    def __init__(self):
        super().__init__()
        self.min = None
        self.max = None


    def fit(self,data:np.ndarray)->None:
        """
        Method to fit the transformer using the min max method.

        Parameters
        ----------
        data: np.ndarray
            Data to extract the min and max values of the dataset.
        """
        self.min = np.min(data)
        self.max = np.max(data)
        super().fit(data)

    def transform(self, data:np.ndarray)->np.ndarray:
        super().transform(data)
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data:np.ndarray)->np.ndarray:
        super().inverse_transform(data)
        return data * (self.max - self.min) + self.min
    
    def get_config(self)->dict[str,float]:
        """
        Method to get the configuration of the instance.
        
        Returns
        -------
        dict[str,float]
            Min and max values of the instance.
        """
        config = super().get_config()
        config.update({"min":self.min, "max":self.max})
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
        self.min = config["min"]
        self.max = config["max"]