from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload

@dataclass
class Optimizer: 
    eta: float
    # gradient_method: str
    gamma: float = None
    minibatch_size: int = None
    lambd: float = 0 
    beta: float = None
    tuning_method: str = 'none'
    tuning_method_options: tuple = field(init=False, default = ('none', 'rms_prop'))


    # gradient_method_options: list = field(init=False, default_factory=lambda: ['gd', 'sgd'])


    # Parameters used for update
    W_change: np.ndarray = field(init=False, default=None) # Previous change in weights
    b_change: np.ndarray = field(init=False, default=None) # Previous change in bias

    # rms prop Parameters
    W_second_momentum: np.ndarray = field(init=False, default=None) # Previous change in weights
    b_second_momentum: np.ndarray = field(init=False, default=None) # Previous change in bias
    epsilon: np.ndarray = field(init=False, default=None) # Previous change in bias


    number_of_updates: int = 0

    def __post_init__(self): 

        if self.gamma != None:
            if not 0 <= self.gamma <= 1:
                raise ValueError('Allowed range for gamma: [0, 1]')
            self.W_change = 0.0
            self.b_change = 0.0

        assert(self.tuning_method in self.tuning_method_options)

            # Inital values
        if self.tuning_method == 'none': 
            pass
        elif self.tuning_method == 'rms_prop': 
            self.__test_beta()
            self.epsilon = 1e-8
            self.W_second_momentum = 0.0
            self.b_second_momentum = 0.0


    def __test_beta(self):
        if self.beta == None: 
            raise ValueError(f'Tuning method {self.tuning_method} requires a value for beta, not None')
        if not 0 <= self.beta < 1: 
            raise ValueError('Allowed range for beta: (0, 1]')


    def update_change(self, gradient_weights, gradient_bias) -> tuple[np.ndarray, np.ndarray]:
        self.number_of_updates += 1

        if self.tuning_method == 'none':
            # Gradient descent
            W_change = self.eta * gradient_weights
            b_change = self.eta * gradient_bias

        elif self.tuning_method == 'rms_prop':
            self.W_second_momentum = self.beta * self.W_second_momentum + (1-self.beta) * gradient_weights**2 
            self.b_second_momentum = self.beta * self.b_second_momentum + (1-self.beta) * gradient_bias**2
            W_change = self.eta * gradient_weights / np.sqrt(self.W_second_momentum + self.epsilon)
            b_change = self.eta * gradient_bias / np.sqrt(self.b_second_momentum + self.epsilon)



        if self.gamma != None: 
            # Momentum
            W_change += self.gamma*self.W_change
            b_change += self.gamma*self.b_change

        if self.lambd > 0:
            # L2 regularization
            W_change += self.lambd * gradient_weights # FIXME: Add to bias ? 
            b_change += self.lambd * gradient_bias


        self.W_change, self.b_change = W_change, b_change

        return W_change, b_change




