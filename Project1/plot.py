import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import sys
import importlib
import franke_data
import ols 
import analysis 
importlib.reload(franke_data)
importlib.reload(ols)
importlib.reload(analysis)

@dataclass
class Plot:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    poly_score: dict # 

    poly_deg: np.ndarray = field(init=False)


    def __post_init__(self):
        assert(np.size(self.X_train) > np.size(self.X_test))
        assert(np.size(self.y_train) > np.size(self.y_test))
        self.poly_deg = np.array([ int(i) for i in self.poly_score.keys() ])

    def plot_mse(self): 
        try:
            keys = self.poly_score['1']['mse'].keys()
        except KeyError:
            print('KeyError: MSe values is not found in poly_score')
            return

        plt.figure()
        for label in keys:
            # list comprehension, iterate through outer keys
            y_axis_plot = np.array([ self.poly_score[str(i)]['mse'][label] for i in self.poly_deg ])
            plt.plot(self.poly_deg, y_axis_plot, label = label)

        plt.xlabel('poly deg')
        plt.ylabel('MSE')
        plt.legend()
        plt.ylim(0,2)
        plt.show()

    def plot_r2(self): 
        try:
            keys = self.poly_score['1']['r2'].keys()
        except KeyError:
            print('KeyError: R2 values is not found in poly_score')
            return

        plt.figure()
        for label in keys:
            # list comprehension, iterate through outer keys
            y_axis_plot = np.array([ self.poly_score[str(i)]['r2'][label] for i in self.poly_deg ])
            plt.plot(self.poly_deg, y_axis_plot, label = label)

        plt.xlabel('poly deg')
        plt.ylabel('R2')
        plt.legend()
        plt.show()

    def plot_beta(self): 
        try:
            keys = self.poly_score['1']['beta'].keys()
        except KeyError:
            print('KeyError: Beta values is not found in poly_score. \n' \
                  'Update dict and re-run constructor.')
            return

        # # beta = [ self.poly_score[str(j)]['beta'][j] ]
        # for i in self.poly_deg:
        #     e = self.poly_score[str(i)]['beta']
        #     print(e)
        #     breakpoint() 


        







        

if __name__ == '__main__':

    max_poly_deg = 8
    n_data = 10000
    test_size = 0.2


    f = franke_data.FrankeData(max_poly_deg, n_data, data_dim = 1, add_noise = 0.5)
    X_train, X_test, y_train, y_test = f.get_train_test_data(test_size = test_size)

    score_list = ['mse', 'r2', 'beta']
    method_list = ['ols_own', 'ols_own']
    data_list = ['train', 'test']

    a = analysis.Analysis(X_train, X_test, y_train, y_test)
    score = a.calculate_loop(max_poly_deg, score_list, method_list, data_list)

    p = Plot(X_train, X_test, y_train, y_test, score)
    p.plot_beta()

