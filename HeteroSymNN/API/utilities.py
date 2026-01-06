import numpy as np
from typing import Any

class DataScaler:
    def __init__(self,data:np.ndarray):
        pass

    def normalize(self, data:np.ndarray)->np.ndarray:
        raise NotImplementedError

    def denormalize(self, data:np.ndarray)->np.ndarray:
        raise NotImplementedError
    
    def get_config(self)->dict[str,Any]:
        raise NotImplementedError
    
    def set_config(self)->None:
        raise NotImplementedError

class MinMaxScaler(DataScaler):
    def __init__(self,data:np.ndarray):
        self.min = min(data)
        self.max = max(data)

    def normalize(self, data:np.ndarray)->np.ndarray:
        return (data - self.min) / (self.max - self.min)

    def denormalize(self, data:np.ndarray)->np.ndarray:
        return data * (self.max - self.min) + self.min
    
    def get_config(self)->dict[str,Any]:
        return {"min":self.min, "max":self.max}
    
    def set_config(self,config:dict[str,Any])->None:
        self.min = config["min"]
        self.max = config["max"]