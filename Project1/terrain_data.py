from dataclasses import dataclass, field
import numpy as np
import imageio
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform
from sklearn.model_selection import  train_test_split
import importlib
import plot_model
importlib.reload(plot_model)

@dataclass
class TerrainData:
    n: int # max poly deg 
    N: int # datapoints

    terrain_file: str = 'SRTM_data_Norway_1.tif'

    test_size: float = 0.2

    # data_dim: int = 2
    set_seed: bool = True
    seed: int = field(init=False, default = 1)

    x: np.ndarray = field(init=False, repr=False) # mesh grid of x cooridiantes
    y: np.ndarray = field(init=False, repr=False) # mesh grid of y cooridiantes
    z: np.ndarray = field(init=False, repr=False) # terrain data
    X: np.ndarray = field(init=False, repr=False)

    X_train: np.ndarray = field(init=False, repr=False)
    X_test: np.ndarray = field(init=False, repr=False)
    y_train: np.ndarray = field(init=False, repr=False)
    y_test: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        x = np.linspace(0, 1, self.N)
        y = np.linspace(0, 1, self.N)
        self.x, self.y = np.meshgrid(x, y)
        self.X = self.__design_matrix(self.x, self.y, self.n)
        self.z = self.__terrain_data()
        self.__split_data()

    def __split_data(self):
        if self.set_seed:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                    train_test_split(self.X, self.z, test_size=self.test_size, random_state=self.seed)
        else: 
            self.X_train, self.X_test, self.y_train, self.y_test = \
                    train_test_split(self.X, self.z, test_size=self.test_size)

    def __terrain_data(self):
        # Read data
        terrain = imageio.v2.imread(self.terrain_file)
        terrain_slice = terrain[:self.N, :self.N]
        return np.ravel(terrain_slice)

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

    def plot_data(self):
        # Load the terrain

        # Show the terrain
        terrain = np.reshape(self.z, (self.N, self.N))
        plt.figure()
        plt.title('Terrain over Norway 1')
        plt.imshow(terrain, cmap='gray')
        plt.xlabel('x')
        plt.ylabel('y')

        terrain_full = imageio.v2.imread(self.terrain_file)
        plt.figure()
        plt.title('Terrain over Norway 1')
        plt.imshow(terrain_full, cmap='gray')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


    def get_train_test_data(self):
        """ returns tuple: (X_train, X_test, y_train, y_test) """
        return self.X_train, self.X_test, self.y_train, self.y_test


    def get_l(self, deg): 
        l = int(((deg+1)*(deg+2)/2))		# Number of elements in beta
        return l 

    def get_X_train(self, deg: int=None): 
        if deg == None: 
            return self.X_train

        return self.X_train[:, :self.get_l(deg)]

    def get_X_test(self, deg: int=None): 
        if deg == None: 
            return self.X_test

        return self.X_test[:, :self.get_l(deg)]


    def get_y_train(self):
        return self.y_train


    def get_y_test(self): 
        return self.y_test


    def get_X(self, deg: int=None):
        if deg == None: 
            return self.X

        return self.X[:, :self.get_l(deg)]


    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z


    def get_test(self, deg: int=None): 
        if deg == None: 
            return self.X_test, self.y_test

        return self.X_test[:, :self.get_l(deg)], self.y_test

    def get_train(self, deg: int=None): 
        if deg == None: 
            return self.X_train, self.y_train

        return self.X_train[:, :self.get_l(deg)], self.y_train

    def get_data(self, deg: int=None):
        if deg == None:
            return self.X, self.z

        return self.X[:, :self.get_l(deg)], self.z


if __name__ == '__main__':
    t = TerrainData(5, 500)

    # a = plot_model.PlotModel(t) 
    # a.plot(method='ols_own', data='train')
    # t.plot_data()

