from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload


@dataclass
class Activation: 
    """ Class with activation functions used in ouput and hidden layers of neural network """ 
    z_l: np.ndarray  # Linear transformation output from layer l 
    func_name: str = 'sigmoid' # Name of activation fuctnion


    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
        # return 1/(1 + jnp.exp(-x))

    def derivative_sigmoid(self, sigm): 
        """ Input values gotten from sigmoid funcitno """
        return sigm * (1 - sigm)

    def get_values(self):
        """ 
        Returns: 
            Values from activation function (1. return value) 
            Values from derivative of activation function (2. return values)
        """ 
        if self.func_name == 'sigmoid': 
            values = self.sigmoid(self.z_l)
            return values, self.derivative_sigmoid(values)
        elif self.func_name == 'none': 
            return self.z_l, np.ones_like(self.z_l)




    

