import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import seaborn as sns
import importlib
import franke_data
import time
import sys
import ipdb
import pandas as pd
import ols 
import bdb
importlib.reload(franke_data)
importlib.reload(ols)
plt.style.use('fivethirtyeight')


@dataclass(frozen=True)
class PlotModel: 
    franke_object: franke_data.FrankeData
    X_train: np.ndarray = field(init=False)
    X_test: np.ndarray = field(init=False)
    y_train: np.ndarray = field(init=False)
    y_test: np.ndarray = field(init=False)

    def __post_init__(self):
        train_test_split = self.franke_object.get_train_test_data()
        object.__setattr__(self, 'X_train', train_test_split[0])
        object.__setattr__(self, 'X_test', train_test_split[1])
        object.__setattr__(self, 'y_train', train_test_split[2])
        object.__setattr__(self, 'y_test', train_test_split[3])

    def __franke_funciton(self, x: np.ndarray, y:np.ndarray):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4


    def plot(self, method: str, data: str, data_dim: int): 

        # if data_dim == 2:
        #     N = np.sqrt(len(x))
        #     x = x.reshape(N, N)
        #     y = y.reshape(N, N)
        #     z = z.reshape(N, N)
        #     z_train_data = self.y_train
        #     z_model = self.predict(X)


        if data_dim == 1: 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # NOTE: Change behvaour with plot call args. 
            self.__1d_plot_data_points('train', fig, ax)
            self.__1d_plot_model(data, method, ax)
            self.__1d_plot_franke_funciton(ax)

        elif data_dim == 2:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            self.__2d_plot_franke_fuctnion(ax)
            self.__2d_plot_model(method, ax)

        else:
            raise ValueError
        #     x = self.x 
        #     y = self.y
        #     z = self.z

        #     if self.data_dim > 1:
        #         z = z.reshape(self.N, self.N)

        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     if self.data_dim == 2:
        #         ax.plot_surface(x, y, z) 
        #     elif self.data_dim == 1:
        #         ax.plot(x, y, z)
        #     plt.show()


        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()


    def __1d_plot_data_points(self, dataset: str, fig, ax):
        if dataset == 'train':
            x_data = self.X_train[:, 1] 
            y_data = self.X_train[:, 2]
            z_data = self.y_train
        elif dataset == 'test': 
            x_data = self.X_test[:, 1]
            y_data = self.X_test[:, 2]
            z_data = self.y_test
        else:
            raise ValueError

        # Plot datapoints used for model prediction 
        sort_index = np.argsort(x_data)
        x = x_data[sort_index]
        y = y_data[sort_index]
        z = z_data[sort_index]

        # Plot z data
        ax.plot(x, y, z, '.', markersize = 5, label = f'{dataset} data')


    def __1d_plot_model(self, dataset: str, model: str, ax): 

        if model == 'ols_own': 
            o = ols.OLS(self.X_train, self.y_train)
            o.ols()

        elif model == 'ols_skl':
            o = ols.OLS(self.X_train, self.y_train)
            o.skl_ols()

        else:
            raise ValueError

        if dataset == 'train':
            z_model = o.predict(self.X_train)
            x_data = self.X_train[:, 1] 
            y_data = self.X_train[:, 2]
        elif dataset == 'test': 
            z_model = o.predict(self.X_test)
            x_data = self.X_test[:, 1]
            y_data = self.X_test[:, 2]
        else:
            raise ValueError

        sort_index = np.argsort(x_data)
        x = x_data[sort_index]
        y = y_data[sort_index]
        z = z_model[sort_index]

        ax.plot(x, y, z, linewidth=2, label = f'{dataset} data prediction with {model}')

    def __1d_plot_franke_funciton(self, ax): 
        x_data = self.X_train[:, 1] 
        y_data = self.X_train[:, 2]
        sort_index = np.argsort(x_data)
        x = x_data[sort_index]
        y = y_data[sort_index]
        z = self.__franke_funciton(x, y) 

        ax.plot(x, y, z, linewidth=1, label = 'Franke fuction')


    def __2d_plot_franke_fuctnion(self, ax): 
        x = self.franke_object.x
        y = self.franke_object.y
        z = self.__franke_funciton(x, y)
        surf = ax.plot_surface(x, y, z, label='Franke function')#, label = 'Franke function')
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d


    def __2d_plot_model(self, model: str, ax): 
        # removed Dataset varalbe, meaningless i prediction of datapoints
        # TODO: remove daset parameters from other methods. 

        # ----------------------------
        if model == 'ols_own': 
            o = ols.OLS(self.X_train, self.y_train)
            o.ols()

        elif model == 'ols_skl':
            o = ols.OLS(self.X_train, self.y_train)
            o.skl_ols()

        else:
            raise ValueError

        # ^^^^^^^^^^^^^^^^^^^^^^   copy from 1D

        # If 2 dim
        z_model = o.predict(self.franke_object.X)
        N = self.franke_object.N
        Z = z_model.reshape(N, N)
        X = self.franke_object.x
        Y = self.franke_object.y

        surf = ax.plot_surface(X, Y, Z, label=f'Model: {model}')#, label = 'Franke function')
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d

        


        # pd.DataFrame(a).to_csv('sample.csv')
        # if self.data_dim > 1:
        #     z = z.reshape(self.N, self.N)
    
        # plt.imsave('matrix.jpeg', x)

        # x_data = self.X_train[:, 1] 
        # y_data = self.X_train[:, 2]
        # x_data = x_data.reshape()
        # y_data = y_data.reshape()
        # print('2d plot franke function')

        # x_data = self.X_train[:, 1] 
        # y_data = self.X_train[:, 2]

        # plot franke function 

    #     print('2d')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # if data_dim == 2:
    #     ax.plot_surface(x, y, z) 
    # elif data_dim == 1:
    #     ax.plot(x, y, z)
    # plt.show()





if __name__ == '__main__': 

    np.random.seed(0)


    max_poly_deg = 1
    n_data = 100
    # n_data = 2000000
    test_size = 0.2
    noise = 0.01
    data_dim = 2

    # XXX:  fix noise plot 
    f = franke_data.FrankeData(max_poly_deg, n_data, data_dim = data_dim, add_noise = noise, test_size = test_size)
    X_train, X_test, y_train, y_test = f.get_train_test_data() # XXX pass to function call

    
    a = PlotModel(f) 
    a.plot(method='ols_own', data='test', data_dim = data_dim)


    # a.plot_model_2d(data_dim = 2)
    # a.plot_model(method = 'ols_own', data = 'train', data_dim = 1)
    # a.plot(method = 'ols_skl', data = 'test', data_dim = 1)
    # a.plot(method = 'ols_own', data = 'test', data_dim = 1)

