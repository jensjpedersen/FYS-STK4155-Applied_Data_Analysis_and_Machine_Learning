import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import seaborn as sns
import importlib
import terrain_data
import time
import sys
import ipdb
import pandas as pd
import ols 
import bdb
importlib.reload(terrain_data)
importlib.reload(ols)
plt.style.use('fivethirtyeight')



@dataclass(frozen=True)
class PlotTerrain: 
    terrain_object: terrain_data.TerrainData

    def plot(self, method: str): 

        fig = plt.figure(figsize=(20,8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        self.__plot_terrain_surface(ax1)
        self.__2d_plot_model(method, ax2, alpha = 1)

        self.__plot_data_points('train', ax1, color = 'y')
        self.__plot_data_points('test', ax1, color = 'm')

        self.__plot_data_points('train', ax2, color = 'y')
        self.__plot_data_points('test', ax2, color = 'm')


            # map(self.__plot_data_points, ax, data_list)
            # self.__plot_data_points(data, ax)

        ax1.set_xlabel('x')
        ax2.set_xlabel('x')
        ax1.set_ylabel('y')
        ax2.set_ylabel('y')
        ax1.set_zlabel('Height [m]')
        ax2.set_zlabel('Height [m]')
        ax1.legend()
        ax2.legend()

        plt.subplots_adjust(wspace=0.1, hspace=0)
        plt.show()


    def __plot_data_points(self, dataset: str, ax, color):
        get_func_name = f'get_{dataset}'
        get_data = getattr(self.terrain_object, get_func_name)
        X_data_terrain, y_data_terrain = get_data()

        x_data = X_data_terrain[:,1]
        y_data = X_data_terrain[:,2]
        z_data = y_data_terrain

        # Plot datapoints used for model prediction 
        sort_index = np.argsort(x_data)
        x = x_data[sort_index]
        y = y_data[sort_index]
        z = z_data[sort_index]

        # Plot z data
        ax.plot(x, y, z, '.', markersize = 5, alpha = 0.5, label = f'{dataset} data', color = color)

    def __plot_terrain_surface(self, ax): 

        X_terrain, y_terrain = self.terrain_object.get_data()
        N = self.terrain_object.N
        x = X_terrain[:, 1]
        y = X_terrain[:, 2]
        z = y_terrain

        if len(np.shape(x)) == 1: 
            x = np.reshape(x, (N,N))

        if len(np.shape(y)) == 1: 
            y = np.reshape(y, (N,N))

        if len(np.shape(z)) == 1: 
            z = np.reshape(z, (N,N))

        # XXX: sorting
        surf = ax.plot_surface(x, y, z, label='Terrain Data', alpha=1)#, label = 'Franke function')
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d


    def __2d_plot_model(self, model: str, ax, alpha): 
        # TODO: in progress
        X_train, y_train = self.terrain_object.get_train()
        if model == 'ols_own': 
            o = ols.OLS(X_train, y_train)
            o.ols()

        elif model == 'ols_skl':
            o = ols.OLS(X_train, y_train)
            o.skl_ols()

        else:
            raise ValueError

        # If 2 dim
        z_model = o.predict(self.terrain_object.get_X_data())
        N = self.terrain_object.N
        Z = z_model.reshape(N, N)
        X = self.terrain_object.x
        Y = self.terrain_object.y

        surf = ax.plot_surface(X, Y, Z, label=f'Model: {model}', alpha = alpha)#, label = 'Franke function')
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

    t = terrain_data.TerrainData(16, 40)
    p = PlotTerrain(t)
    p.plot( method='ols_own', data_list = ['test', 'train'])



