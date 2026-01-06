from __future__ import annotations
from typing import Optional,Literal,Union,Any
import warnings
import numpy as np

from ...Backend import hardware as HW
from ...types import NodeConfig,LayerValues,LayerConstructionConfig,FlexibleNodeConfig,BackendArray,ConstantToUpdate
from ..layers import Layer
from .base_classes import BaseNetwork
from .. import losses as lossC, optimizers as OptiC, initializers as InitC

class EvoNet(BaseNetwork):
    def __init__(self):
        super().__init__()