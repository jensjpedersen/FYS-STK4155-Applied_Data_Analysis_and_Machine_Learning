from dataclasses import dataclass, field
# import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import poly_data
import autograd.numpy as np  # Thinly-wrapped numpy
# import jax.numpy as np 
from autograd import grad 
import pprint


@dataclass()
class GradientDescent: 
    data_object: object  

    dataset:str = 'train'

    X_data: np.ndarray = field(init=False, repr=False) # X data with resepct to dataset variable
    y_data: np.ndarray = field(init=False, repr=False) # y --||--

    thetas: np.ndarray = field(init=False, repr=False) # Array with thetas for all iterations
    thetas_init: np.ndarray = field(init=False) # Intial condition for thata

    def __post_init__(self):
        get_func_name = f'get_{self.dataset}'
        get_data = getattr(self.data_object, get_func_name)
        self.X_data, self.y_data= get_data()
        self.set_initial_conditions()
    
    def costOLS(self, X, y, theta):
        return np.sum((y[:,np.newaxis] - X @ theta)**2)

    def grad_costOLS(self): 
        return 

    def get_hessian(self): 
        pass

    def set_initial_conditions(self,):
        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        self.thetas_init = np.random.randn(n_coeff, 1) # Inital guess for theta's 

    def plane_gd(self, eta: float, n_epochs: int = 1000): 
        """ 
        eta = learning rate
        """
        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        theta_new = np.random.randn(n_coeff, 1) # Inital guess for theta's 
        theta_new = self.thetas_init

        thetas = np.zeros((n_epochs, n_coeff))

        for i in range(n_epochs): 
            # Change to while wiht tolearnace
            theta_old = theta_new

            # Claculate gradient
            try: 
                grad_func = grad(self.costOLS, 2)
            except: 
                pass
            gradients = grad_func(self.X_data, self.y_data, theta_new)
            theta_new = theta_old - eta*gradients
            thetas[i,:] = theta_new.T

        self.thetas = thetas

    def gd_momentum(self, eta: float, gamma: float, n_epochs: int = 1000):
        """ 
        eta = learning rate
        gamma = momentum 
        """

        if not 0 <= gamma <= 1:
            raise ValueError('Allowed range for gamma: [0, 1]')


        # Initial values 
        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        theta_new = np.random.randn(n_coeff, 1) # Inital guess for theta's 
        theta_new = self.thetas_init
        change = 0.0

        thetas = np.zeros((n_epochs, n_coeff))


        for i in range(n_epochs): 
            # Change to while wiht tolearnace
            theta_old = theta_new

            # Claculate gradient
            try: 
                grad_func = grad(self.costOLS, 2)
            except: 
                pass

            gradients = grad_func(self.X_data, self.y_data, theta_new)
            # theta_new = eta*gradients
            new_change = eta*gradients + gamma*change
            theta_new = theta_old - new_change
            thetas[i,:] = theta_new.T

            change = new_change

        self.thetas = thetas

    def sgd(self, eta: float, size_batch: int, n_epochs: int = 100):

        n_data = len(self.y_data)
        n_minibatches = int(n_data/size_batch)

        print(f'n_data = {n_data}')
        print(f'n_minibatches = {n_minibatches}')
        print(f'n_epochs = {n_epochs}')
        print(f'size_batch = {size_batch}')
        

        for epoch in range(n_epochs): 
            for i in range(n_minibatches): 
                k = np.random.randint(n_minibatches) # Pick random minibatch
                slice_0 = k*size_batch
                slice_1 = (k+1)*size_batch 
                minibatch_X = self.X_data[slice_0:slice_1]
                minibatch_y = self.y_data[slice_0:slice_1]

                print(f'k = {k}')
                print(f'minibatch_X = {minibatch_X}')
                print(f'minibatch_y = {minibatch_y}')


        print(f'self.X_data = {self.X_data}')
        print(f'self.y_data = {self.y_data}')
        




    def get_theta(self): 
        return self.thetas[-1, :]

    def get_thetas(self):
        return self.thetas















