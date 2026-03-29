import numpy as np
from typing import Any

class DataScaler:
    """
    Base class for data scalers.

    Parameters
    ----------
    data : np.ndarray
        Data to extract the parameters nessesary for the scaler.
    """
    def __init__(self,data:np.ndarray):
        pass

    def normalize(self, data:np.ndarray)->np.ndarray:
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
        """
        raise NotImplementedError

    def denormalize(self, data:np.ndarray)->np.ndarray:
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
        """
        raise NotImplementedError
    
    def get_config(self)->dict[str,Any]:
        """
        Method to get the configuration of the scaler.

        Needs to return the values of the internal parameters of the scaler for it to be able to reconstruct itself with out needing the data.
        
        Returns
        -------
        dict[str,Any]
            Configuration of the scaler.
        """
        raise NotImplementedError
    
    def set_config(self,config:dict[str,Any])->None:
        """
        Method to set the configuration of the scaler.

        From a dictionary with strings as keys be able to reconstruct the scaler with out needing to pass the data.

        Parameters
        ----------
        config : dict[str,Any]
            Configuration of the scaler.
        """
        raise NotImplementedError

class MinMaxScaler(DataScaler):
    """
    Scaler class that normalizes data between 0 and 1 with a min and max value method.

    Parameters
    ----------
    data : np.ndarray
        Data to extract min and max of the values is going to transform.
    """
    def __init__(self,data:np.ndarray):
        self.min = min(data)
        self.max = max(data)

    def normalize(self, data:np.ndarray)->np.ndarray:
        return (data - self.min) / (self.max - self.min)

    def denormalize(self, data:np.ndarray)->np.ndarray:
        return data * (self.max - self.min) + self.min
    
    def get_config(self)->dict[str,float]:
        """
        Method to get the configuration of the scaler.
        
        Returns
        -------
        dict[str,float]
            Min and max values of the scaler.
        """
        return {"min":self.min, "max":self.max}
    
    def set_config(self,config:dict[str,float])->None:
        """
        Method to set the configuration of the scaler.
        
        Parameters
        ----------
        config : dict[str,float]
            Min and max values of the scaler.
        """
        self.min = config["min"]
        self.max = config["max"]