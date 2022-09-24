# Common imports
from dataclasses import dataclass, field
import os
import sys
import numpy as np
from numba import njit
import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

@dataclass
class FrankeData: 
    """ Class for generating data from franke funciton

    Args:
        N (int): Number of datapoints
        n (str): Polynomial degree
        add_noise (bool):  if true adds mu(0, 1) noise to franke fucntion
        uniform_data (bool): if true, creates uniform data in x and y. 
        data_dim (int): Number of dimensions on dataset. Values: 1 or 2 (meshgrid). 
        x_range (list[int]): Range of x values, [x_min, x_max]
        y_range (list[int]): Range of y values, [y_min, y_max] 
    """
    n: int
    N: int

    add_noise: float = 0
    uniform_data: bool = True
    data_dim: int = 2
    test_size: float = 0.2
    

    x_range: list[int] = field(default_factory = lambda: [0, 1])
    y_range: list[int] = field(default_factory = lambda: [0, 1])

    x: np.ndarray = field(init=False, repr=False) 
    y: np.ndarray = field(init=False, repr=False)
    z: np.ndarray = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)

    X_train: np.ndarray = field(init=False, repr=False)
    X_test: np.ndarray = field(init=False, repr=False)
    y_train: np.ndarray = field(init=False, repr=False)
    y_test: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.x, self.y = self.__generate_xy_data(self.uniform_data, self.data_dim)
        self.z = self.__franke_funciton(self.x, self.y, self.add_noise)
        self.X = self.__design_matrix(self.x, self.y, self.n)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.z, test_size=self.test_size)

    def get_train_test_data(self):
        """ returns tuple: (X_train, X_test, y_train, y_test) """
        return self.X_train, self.X_test, self.y_train, self.y_test


    def get_l(self, deg): 
        l = int(((deg+1)*(deg+2)/2))		# Number of elements in beta
        return l 

    def get_X_train(self, deg: int=None): 
        if deg == None: 
            return self.X_train

        return self.X_train[:, :self.get_l(deg)-1]

    def get_X_test(self, deg: int=None): 
        if deg == None: 
            return self.X_test

        return self.X_test[:, :self.get_l(deg)-1]


    def get_y_train(self):
        return self.y_train


    def get_y_test(self): 
        return self.y_test

    def get_X_vali(self, deg: int=None):
        pass

    def get_y_val(self): 
        pass

    def get_X(self, deg: int=None):
        if deg == None: 
            return self.X

        return self.X[:, :self.get_l(deg)-1]


    def get_y(self):
        return self.y


    def print_design_matrix(self): 
        n = self.n 
        print(f"X[:,0] = 1")
        for i in range(1,n+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                # X[:,q+k] = (x**(i-k))*(y**k)
                print(f"X[:,{q+k}] = x^{i-k} y^{k}")

    def plot(self):
        x = self.x 
        y = self.y
        z = self.z

        if self.data_dim > 1:
            z = z.reshape(self.N, self.N)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if self.data_dim == 2:
            ax.plot_surface(x, y, z) 
        elif self.data_dim == 1:
            ax.plot(x, y, z)
        plt.show()


    def __generate_xy_data(self, uniform: bool, dim: int):
        if uniform: 
            x = np.sort(np.random.uniform(self.x_range[0], self.x_range[1], self.N))
            y = np.sort(np.random.uniform(self.x_range[0], self.x_range[1], self.N))
        else:
            x = np.linspace(self.x_range[0], self.x_range[1], self.N)
            y = np.linspace(self.y_range[0], self.y_range[1], self.N)

        if dim == 1:
            # Constant y values. Centered with respect to y_range 
            y = np.ones_like(x) * (self.y_range[1] - self.y_range[0])/2
        elif dim == 2:
            x, y = np.meshgrid(x,y)
        else:
            raise ValueError('data_dim is either 1 or 2')

        return x, y 

    def __franke_funciton(self, x: np.ndarray, y: np.ndarray, add_noise: float):
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

        if add_noise > 0: 
            noise = add_noise*np.random.normal(0, 1, np.shape(x))
            return term1 + term2 + term3 + term4 + noise

        return term1 + term2 + term3 + term4

    def __design_matrix(self, x: np.ndarray, y: np.ndarray, n: int):
        """ n (int) - Polynomial degree """ 
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int(((n+1)*(n+2)/2))		# Number of elements in beta

        X = np.ones((N,l))
        X = self.__jit_design_matrix_loop(x,y,n,X)
        return X


    @staticmethod
    @njit
    def __jit_design_matrix_loop(x, y, n, X):
        for i in range(1,n+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)
        return X



if __name__ == '__main__':
    tic = time.perf_counter()

    f = FrankeData(n=6, N=1000, add_noise = False, data_dim = 2)
    toc = time.perf_counter()
    print(f'took: {toc-tic}s')
    X_train, X_test, y_train, y_test = f.get_train_test_data()

    f.plot()


    # Making meshgrid of datapoints and compute Franke's function
    sys.exit()
    n = 5
    N = 1000
    z = FrankeFunction(x, y)
    X = create_X(x, y, n=n)    
    # split in training and test data
    X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.2)

    clf = skl.LinearRegression().fit(X_train, y_train)

    # The mean squared error and R2 score
    print("MSE before scaling: {:.2f}".format(mean_squared_error(clf.predict(X_test), y_test)))
    print("R2 score before scaling {:.2f}".format(clf.score(X_test,y_test)))

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Feature min values before scaling:\n {}".format(X_train.min(axis=0)))
    print("Feature max values before scaling:\n {}".format(X_train.max(axis=0)))

    print("Feature min values after scaling:\n {}".format(X_train_scaled.min(axis=0)))
    print("Feature max values after scaling:\n {}".format(X_train_scaled.max(axis=0)))

    clf = skl.LinearRegression().fit(X_train_scaled, y_train)


    print("MSE after  scaling: {:.2f}".format(mean_squared_error(clf.predict(X_test_scaled), y_test)))
    print("R2 score for  scaled data: {:.2f}".format(clf.score(X_test_scaled,y_test)))
