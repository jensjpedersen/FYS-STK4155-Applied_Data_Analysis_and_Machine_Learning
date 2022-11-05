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
    gradient_method: str
    gamma: float = None
    minibatch_size: int = None

    gradient_method_options: list = field(init=False, default_factory=lambda: ['gd', 'sgd'])


    # Parameters used for update
    W_change: np.ndarray = field(init=False, default=None) # Previous change in weights
    b_change: np.ndarray = field(init=False, default=None) # Previous change in bias

    def __post_init__(self): 

        if self.gamma != None:
            if not 0 <= self.gamma <= 1:
                raise ValueError('Allowed range for gamma: [0, 1]')
            self.W_change = 0.0
            self.b_change = 0.0


    def update_change(self, gradient_weights, gradient_bias) -> tuple[np.ndarray, np.ndarray]:

        # TODO: Update sheme
        if self.gamma != None: 
            W_change = self.eta * gradient_weights + self.gamma*self.W_change
            b_change = self.eta * gradient_bias + self.gamma*self.b_change
        else: 
            W_change = self.eta * gradient_weights
            b_change = self.eta * gradient_bias

        self.W_change, self.b_change = W_change, b_change

        return W_change, b_change




