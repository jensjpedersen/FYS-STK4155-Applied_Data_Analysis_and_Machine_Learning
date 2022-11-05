from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import logging

# logging.basicConfig(filename='flow.log', encoding='utf-8', level=logging.DEBUG)

@dataclass
class Activation: 
    """ Class with activation functions used in ouput and hidden layers of neural network """ 
    z_l: np.ndarray  # Linear transformation output from layer l 
    func_name: str # Name of activation fuctnion
    options: list = field(default_factory=lambda: ['sigmoid', 'none'])

    def __post_init(self):
        assert(func_name in self.options)



    def __sigmoid(self, x):
        return 1/(1 + np.exp(-x))
        # return 1/(1 + jnp.exp(-x))

    def __derivative_sigmoid(self, sigm): 
        """ Input values gotten from sigmoid funcitno """
        return sigm * (1 - sigm)

    def get_values(self):
        """ 
        Returns: 
            Values from activation function (1. return value) 
            Values from derivative of activation function (2. return values)
        """ 
        if self.func_name == 'sigmoid': 
            values = self.__sigmoid(self.z_l)
            return values, self.__derivative_sigmoid(values)
        elif self.func_name == 'none': 
            # return self.z_l, np.ones_like(self.z_l)
            # return 1, 1
            return self.z_l, np.ones_like(self.z_l)
        else:
            raise ValueError(f'Valid options for func_name are: {self.options}, got {self.func_name}')




    

