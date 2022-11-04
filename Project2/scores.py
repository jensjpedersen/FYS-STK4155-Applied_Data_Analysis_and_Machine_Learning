from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import jax.numpy as jnp 
from jax import grad

# def MSE(y, t):
#     """ Squared error cost function""" # XXX: removed 1/2
#     # assert(len(y.shape) == 1)
#     return jnp.sum((t - y)**2)

@dataclass
class Scores:
    """


    """
    a_L: np.ndarray = field(repr=False) # Ouput from output layer
    t: np.ndarray = field(repr=False) # Targets
    score_name: str = 'mse'
    options: list = field(init=False, default_factory=lambda: ['mse'])


    def __post_init(self):
        assert(self.score_name in self.options)

    def MSE(self, y, t):
        """ Squared error cost function""" # XXX: removed 1/2
        # assert(len(y.shape) == 1)
        return 1/2 * np.sum((t - y)**2)

    # def MSE(self, y, t):
    #     """ Squared error cost function""" # XXX: removed 1/2
    #     # assert(len(y.shape) == 1)
    #     # return 1/2 * jnp.sum((t - y)**2)


    def get_score(self): 
        if self.score_name == 'mse': 
            return float(self.MSE(self.a_L, self.t))
        else:
            raise ValueError('Allowed opitons for score_name is: {self.options}')

    def get_derivative(self): 
        if self.score_name == 'mse': 
            # return grad(self.MSE, 1)(self.a_L, self.t) # XXX: 0 is correct?
            # return grad(self.MSE, 1)(self.a_L, self.t) # XXX: 0 is correct?
            return self.a_L - self.t
        else:
            raise ValueError('Allowed opitons for score_name is: {self.options}')


