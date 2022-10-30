from dataclasses import dataclass, field
# import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import poly_data
# import autograd.numpy as np  # Thinly-wrapped numpy
import numpy as np
import jax.numpy as jnp 
from jax import grad
# import jax.numpy as np 
# from autograd import grad 
import pprint
import time


def log_time(func):
    def wrapper(*args, **kwargs):
        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        print(f'Function {func.__name__!r} executed in {(toc-tic):.4f}s')
        return result

    return wrapper


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
        assert(len(y) != 1)
        return jnp.sum((y[:,np.newaxis] - X @ theta)**2)/len(y) # XXX

    def grad_costOLS(self): 
        return 

    def get_hessian(self): 
        pass

    def set_initial_conditions(self,):
        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        self.thetas_init = np.random.randn(n_coeff, 1) # Inital guess for theta's 
        print(self.thetas_init)


    def delta_gd(self, X, y, theta, eta, momentum = None): 
        grad_func = grad(self.costOLS, 2)
        gradients = grad_func(X, y, theta)

        if momentum == None:
            return -eta * gradients


    
        

    def update_adagrad(self, X, y, theta):
        pass


    def plane_gd(self, eta: float, n_epochs: int = 100, tune_method: str = None): 
        """ 
        eta = learning rate

        tune_method = None (constant learning rate), adagrad, rmsprop, adam
        """
        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        theta_new = self.thetas_init # Intial guess for thetas
        print(f'theta_new = {theta_new}')
        

        thetas = np.zeros((n_epochs+1, n_coeff))
        thetas[0,:] = theta_new.T


        for i in range(1, n_epochs+1): 
            # Change to while wiht tolearnace
            theta_old = theta_new

            # Update func returns change
            # Claculate gradient
            delta = self.delta_gd(self.X_data, self.y_data, theta_new, eta)
            theta_new = theta_old + delta
            thetas[i,:] = theta_new.T

        self.thetas = thetas


    def gd_momentum(self, eta: float, gamma: float, n_epochs: int = 100):
        """ 
        eta = learning rate
        gamma = momentum 
        """

        if not 0 <= gamma <= 1:
            raise ValueError('Allowed range for gamma: [0, 1]')


        # Initial values 
        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        theta_new = self.thetas_init
        change = 0.0

        thetas = np.zeros((n_epochs+1, n_coeff))
        thetas[0,:] = theta_new.T

        for i in range(1, n_epochs+1): 
            # Change to while wiht tolearnace
            theta_old = theta_new

            # Claculate gradient
            grad_func = grad(self.costOLS, 2)

            gradients = grad_func(self.X_data, self.y_data, theta_new)
            # theta_new = eta*gradients
            new_change = eta*gradients + gamma*change
            theta_new = theta_old - new_change
            thetas[i,:] = theta_new.T

            change = new_change

        self.thetas = thetas

    def new(self, eta: float, n_epochs: int = 100, gamma: float = None):
        """ 
        eta = learning rate
        gamma = momentum 
        """

        if not 0 <= gamma <= 1:
            raise ValueError('Allowed range for gamma: [0, 1]')


        # Initial values 
        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        theta_new = self.thetas_init
        change = 0.0

        thetas = np.zeros((n_epochs+1, n_coeff))
        thetas[0,:] = theta_new.T

        for i in range(1, n_epochs+1): 
            # Change to while wiht tolearnace
            theta_old = theta_new

            # Claculate gradient
            grad_func = grad(self.costOLS, 2)

            gradients = grad_func(self.X_data, self.y_data, theta_new)
            # theta_new = eta*gradients
            new_change = eta*gradients + gamma*change
            theta_new = theta_old - new_change
            thetas[i,:] = theta_new.T

            change = new_change

        self.thetas = thetas

    def sgd(self, eta: float, size_batch: int, n_epochs: int = 100):

        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        theta_new = self.thetas_init # Intial guess for thetas


        n_data = len(self.y_data)
        n_minibatches = int(n_data/size_batch)

        thetas = np.zeros((n_epochs*n_minibatches+1, n_coeff))
        thetas[0,:] = theta_new.T
        j = 1


        for epoch in range(1, n_epochs+1): 
            for i in range(n_minibatches): 
                k = np.random.randint(n_minibatches) # Pick random minibatch
                slice_0 = k*size_batch
                slice_1 = (k+1)*size_batch 
                # XXX: Each batch is predifiend
                # XXX: Same minibatch may be selected twice
                minibatch_X = self.X_data[slice_0:slice_1]
                minibatch_y = self.y_data[slice_0:slice_1]

                #XXX Thata updated for each i or only epochs

                # Change to while wiht tolearnace
                theta_old = theta_new

                # Claculate gradient
                grad_func = grad(self.costOLS, 2)
                gradients = grad_func(minibatch_X, minibatch_y, theta_new)
                theta_new = theta_old - eta*gradients
                thetas[j,:] = theta_new.T

                j+=1

        self.thetas = thetas


    def sgd_momentum(self, eta: float, gamma: float, size_batch: int, n_epochs: int = 100):

        if not 0 <= gamma <= 1:
            raise ValueError('Allowed range for gamma: [0, 1]')


        # Initial values 
        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        theta_new = self.thetas_init # Intial guess for thetas
        change = 0.0


        n_data = len(self.y_data)
        n_minibatches = int(n_data/size_batch)

        thetas = np.zeros((n_epochs*n_minibatches+1, n_coeff))
        thetas[0,:] = theta_new.T
        j = 1


        for epoch in range(1, n_epochs+1): 
            for i in range(n_minibatches): 
                k = np.random.randint(n_minibatches) # Pick random minibatch
                slice_0 = k*size_batch
                slice_1 = (k+1)*size_batch 
                # XXX: Each batch is predifiend
                # XXX: Same minibatch may be selected twice
                minibatch_X = self.X_data[slice_0:slice_1]
                minibatch_y = self.y_data[slice_0:slice_1]

                #XXX Thata updated for each i or only epochs

                # Change to while wiht tolearnace
                theta_old = theta_new

                # Claculate gradient
                grad_func = grad(self.costOLS, 2)
                gradients = grad_func(minibatch_X, minibatch_y, theta_new)

                new_change = eta*gradients + gamma*change
                theta_new = theta_old - new_change
                thetas[j,:] = theta_new.T

                change = new_change
                j+=1

        self.thetas = thetas


        
    def get_theta(self): 
        return self.thetas[-1, :]

    def get_thetas(self):
        return self.thetas















