from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
import time
import pandas as pd
import sys
import pprint
import sys
import seaborn as sns
import sklearn as skl
from sklearn.preprocessing import StandardScaler
import math
from mpl_toolkits import mplot3d
import read_csv

class color: 
    red   = "\033[1;31m"
    blue  = "\033[1;34m"
    green = "\033[0;32m" 



@dataclass(frozen=True)
class HeatmapData: 
    """ Object to store heatmap data """
    matrix: np.ndarray
    rows: list 
    columns: list 
    title: str = ""
    info: str = ""

    # Parameters used for styling
    _threshold: float = None
    _fmt: str = '.1f'

    # def __post_init__(self): 
    #     object.__setattr__(self, 'columns', self.df.columns)
    #     object.__setattr__(self, 'rows', self.df.index)

    def get(self) -> tuple[pd.DataFrame, str, str]: 
        return self.df, self.title, self.info

    def plot(self): 
        sns.set_style("darkgrid")
        sns.heatmap(self.df, annot=True, fmt=self._fmt)
        plt.title(f'{self.title}')
        plt.show()


    def print(self): 
        [ print(f'{e[-10:]:10s}', end=' ') for e in self.columns ]

        for j, row in enumerate(self.rows):
            print()
            for i, col in enumerate(self.columns): 
                val = self.matrix[j,i]
                print(f'{val:10.2f}', end=' ')



                # if val < self._threshold: 
                #     print(color.green())



        pass


@dataclass(frozen=True)
class AnalyseCSV: 
    """
    Convernsions: 
        Sorting is done with respect to first series
    """
    readcsv_object: read_csv.ReadCSV 

    #  ____  _       _   _   _             
    # |  _ \| | ___ | |_| |_(_)_ __   __ _ 
    # | |_) | |/ _ \| __| __| | '_ \ / _` |
    # |  __/| | (_) | |_| |_| | | | | (_| |
    # |_|   |_|\___/ \__|\__|_|_| |_|\__, |
    #                                |___/ 

    def __post_init__(self):
        sns.set_style("darkgrid")

    def list_groups(self):
        self.readcsv_object.list_groups()

    def get_group(self, index): 
        return self.readcsv_object.get_group(index)

    def plot_feature_vs_feature(self, index1, index2): 
        feature1 = self.readcsv_object.get_group(index1)
        feature2 = self.readcsv_object.get_group(index2)

        assert(feature1.series_names.all() == feature2.series_names.all())
        assert(feature1.voi_names.all() == feature2.voi_names.all())

        series_names = feature1.series_names
        voi_names = feature1.voi_names


        plt.figure()
        for i, series in enumerate(series_names): 
            sns.scatterplot(x = feature1.feature_matrix[:,i], y = feature2.feature_matrix[:,i])

        plt.xlabel(feature1.feature_name)
        plt.ylabel(feature2.feature_name)


    def plot_feature_vs_feature_subplot(self, index = 8):
        """
        X axis is soreted with respest to 1. series with values corresponding to arg: index (defulat = 8)
            index 8 is shape volume

        ./Examples/plot_feature_vs_feature_subplot.png
        """
        feature = self.get_group(index)
        # assert(feature.feature_name == 'shape volume')
        # idx = np.argsort(np.mean(feature.feature_matrix, axis = 1))
        idx = np.argsort(feature.feature_matrix[:, 0])
        # sorted_matrix = feature1.feature_matrix[idx,:]
        x_values = feature.feature_matrix[idx,:]
        x_label = feature.feature_name


        n = self.readcsv_object.n_features
        cols = 5
        rows = math.ceil(n/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16,8))
        axes = axes.ravel()

        for i in range(n): 
            feature = self.get_group(i)
            sorted_vals = feature.feature_matrix[idx, :]
            ax = axes[i]
            ax.set_ylabel(feature.feature_name)
            ax.set_xlabel(x_label)

            for s, series in enumerate(feature.series_names):
                sns.lineplot(x=x_values[:,s], y=sorted_vals[:,s], label=series,
                             ax=ax)

        plt.legend()
        plt.show()
        

    def plot_feature_vs_voi(self, index1, sort = False):
        """
        ./Examples/plot_features_vs_voi.png
        """
        # TODO: sort values 
        feature1 = self.readcsv_object.get_group(index1)

        voi_names = feature1.voi_names
        x_axis = np.arange(1, len(voi_names)+1)

        feature_matrix = feature1.feature_matrix
        if sort == True: 
            # Sort matrix with respect to 1. series (coloumn 0)
            idx = np.argsort(feature_matrix[:, 0])
            feature_matrix = feature_matrix[idx,:]

        for i, series in enumerate(feature1.series_names): 
            sns.scatterplot(x=x_axis, y=feature_matrix[:,i], label=series)

        plt.xticks(x_axis, voi_names, rotation = 45)
        plt.legend()
        plt.ylabel(feature1.feature_name)
        # plt.show()

    def plot_feature_vs_volume(self, index1):
        # Volume is inex 8 for now
        volume = self.get_group(8)
        assert(volume.feature_name == 'shape volume')
        # Sort with respect to mean volume

        idx = np.argsort(np.mean(volume.feature_matrix, axis = 1))
        sorted_volume_matrix = volume.feature_matrix[idx,:]

        feature1 = self.get_group(index1)
        sorted_feature_matrix = feature1.feature_matrix[idx,:]

        plt.figure()
        for i, series in enumerate(feature1.series_names): 
            sns.lineplot(x=sorted_volume_matrix[:,i], y=sorted_feature_matrix[:,i], label=series)

        plt.legend()
        plt.xlabel(volume.feature_name)
        plt.ylabel(feature1.feature_name)


    def plot_series_vs_mean_subplots(self): 

        n = self.readcsv_object.n_features
        cols = 5
        rows = math.ceil(n/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16,8))
        axes = axes.ravel()

        for i in range(n): 

            feature = self.readcsv_object.get_group(i)
            
            idx = np.argsort(feature.feature_matrix[:, 0])
            sorted_matrix = feature.feature_matrix[idx,:]

            # Plot 1. series (col 0) agains others [:, 1:]
            series_names = feature.series_names
            x_values = np.mean(sorted_matrix, axis = 1)

            n_series = sorted_matrix.shape[1]

            ax = axes[i]

            for s in range(n_series): 
                sns.lineplot(x=x_values, y=sorted_matrix[:, s], label=f'{series_names[s]} VS Series Mean',
                             ax = ax)

                ax.set_xlabel(feature.feature_name)
                ax.set_ylabel(feature.feature_name)

        plt.legend()
        plt.show()


    def plot_series_vs_mean(self, index1): #, sort = False): 
        """ Correlation between series 
        ./Examples/plot_mean_vs_series.png
        """

        #Sort by mean 

        feature1 = self.readcsv_object.get_group(index1)
        
        idx = np.argsort(feature1.feature_matrix[:, 0])
        sorted_matrix = feature1.feature_matrix[idx,:]

        # Plot 1. series (col 0) agains others [:, 1:]
        series_names = feature1.series_names
        x_values = np.mean(sorted_matrix, axis = 1)

        n_series = sorted_matrix.shape[1]

        plt.figure()
        for i in range(0, n_series): 
            sns.lineplot(x=x_values, y=sorted_matrix[:, i], label=f'{series_names[i]} VS Series Mean')

        plt.xlabel(feature1.feature_name)
        plt.ylabel(feature1.feature_name)
        plt.legend()


    def plot_cluster_ravel(self, corr=False): 
        # self.get_group()
        n_features = self.readcsv_object.n_features
        voi_list = list(self.readcsv_object.voi_list)
        series_list = list(self.readcsv_object.series_list)

        matrix = np.zeros((len(voi_list)*len(series_list), n_features))

        feature_names = []

        for i in range(n_features): 
            feature = self.get_group(i)
            matrix[:,i] = feature.feature_matrix.ravel('F')
            feature_names.append(feature.feature_name)

        df = pd.DataFrame(matrix)
        # df.index = voi_list
        df.columns = feature_names

        if corr == True: 
            g = sns.clustermap(df.corr(), 
                       figsize=(8,8), 
                       cbar_pos=(.02, .32, .03, .2),
                       yticklabels=True,
                       xticklabels=True)
            g.ax_row_dendrogram.remove()

        else: 
            pass
            # TODO: add voi + series to label
            # g = sns.clustermap(df,
            #            cbar_pos=(.02, .32, .03, .2),
            #            z_score = 1,
            #            yticklabels=True,
            #            xticklabels=True)

        plt.show()

    def plot_cluster_with_series(self, corr=False): 
        """ 
        corr = True: 
        ./Examples/plot_cluster_with_series_corr.png

        corr = False:
        ./Examples/plot_cluster_with_series.png
        """

        n_features = self.readcsv_object.n_features
        voi_list = list(self.readcsv_object.voi_list)
        series_list = list(self.readcsv_object.series_list)
        n_series = len(series_list)

        matrix = np.zeros((len(voi_list), n_features*n_series))

        feature_names = []

        x_labels = []

        for i in range(n_features): 
            feature = self.get_group(i)
            matrix[:,i*n_series:(i+1)*n_series] = feature.feature_matrix

            feature_names.append(feature.feature_name)
            x_labels.extend([ f'{feature.feature_name} {series}' for series in series_list ])

        df = pd.DataFrame(matrix)
        df.index = voi_list
        df.columns = x_labels

        sns_palette = sns.color_palette("viridis", n_features)
        palette = []
        for color in sns_palette:
            palette.extend([color]*n_series)

        if corr == True: 
            g = sns.clustermap(df.corr(),
                       cbar_pos=(.02, .32, .03, .2),
                       col_colors = palette,
                       yticklabels=True,
                       xticklabels=True)
            g.ax_row_dendrogram.remove()

        else: 
            g = sns.clustermap(df,
                       cbar_pos=(.02, .32, .03, .2),
                       col_colors = palette,
                       z_score = 1,
                       yticklabels=True,
                       xticklabels=True)

        # plt.show()

    def plot_cluster_mean(self, corr = False): 
        """ 
        corr = True:
        ./Examples/plot_cluster_mean_corr.png

        corr = False
        ./Examples/plot_cluster_mean.png
        """
        # self.get_group()
        n_features = self.readcsv_object.n_features
        voi_list = list(self.readcsv_object.voi_list)
        series_list = list(self.readcsv_object.series_list)

        matrix = np.zeros((len(voi_list), n_features))

        feature_names = []

        for i in range(n_features): 
            feature = self.get_group(i)
            matrix[:,i] = np.mean(feature.feature_matrix, axis = 1)
            feature_names.append(feature.feature_name)

        df = pd.DataFrame(matrix)
        df.index = voi_list
        df.columns = feature_names

        if corr == True: 
            g = sns.clustermap(df.corr(),
                       cbar_pos=(.02, .32, .03, .2),
                       yticklabels=True,
                       xticklabels=True)
            g.ax_row_dendrogram.remove()

        else: 
            g = sns.clustermap(df,
                       cbar_pos=(.02, .32, .03, .2),
                       z_score = 1,
                       yticklabels=True,
                       xticklabels=True)

        # plt.show()

    def plot_pair(self, indices:list) -> None: 
        assert(len(indices) > 1)

        df = pd.DataFrame()
        for i in indices: 
            feature = self.get_group(i)
            values = feature.feature_matrix
            values = values.ravel('F')
            n_vois = len(feature.voi_names)
            df[feature.feature_name] = values

        # Create class labels
        classes = [] 
        for series in list(feature.series_names): 
            classes.extend([series]*n_vois)

        df['classes'] = classes

        sns.pairplot(df, hue='classes')

        plt.show()

    def plot_3d(self, indices=()): 
        fig = plt.figure(figsize = (19,9))
        ax = plt.axes(projection = "3d")
        assert(len(indices) == 3)


        n_series = len(self.readcsv_object.series_list)

        for i in range(n_series): 
            x_feature = self.get_group(indices[0])
            y_feature = self.get_group(indices[1])
            z_feature = self.get_group(indices[2])
            x = x_feature.feature_matrix[:,i]
            y = y_feature.feature_matrix[:,i]
            z = z_feature.feature_matrix[:,i]

            ax.scatter3D(x, y, z)





    def get_heatmap_feature_distance(self, index1):
        feature1 = self.readcsv_object.get_group(index1)
        # feature2 = self.readcsv_object.get_group(index2)
        series = feature1.series_names
        mean_matrix = np.zeros((len(series), len(series)))
        std_matrix = np.zeros_like(mean_matrix) 

        val = feature1.feature_matrix

        for i, serie in enumerate(series): 
            repeat_val = np.repeat(feature1.feature_matrix[:,i], len(series))
            repeat_val = np.reshape(repeat_val, (-1, len(series)))
            diff = val - repeat_val
            mean = np.mean(diff, axis=0)
            std = np.std(diff, ddof=1, axis=0)
            mean_matrix[i,:] = mean 
            std_matrix[i,:] = std


        cov_matrix = std_matrix/mean_matrix*100
        heatmap_mean = HeatmapData(mean_matrix, series, series, feature1.feature_name)
        heatmap_std = HeatmapData(std_matrix, series, series, feature1.feature_name)
        heatmap_cov = HeatmapData(cov_matrix, series, series, feature1.feature_name)
        return heatmap_mean, heatmap_std, heatmap_cov










       


    


