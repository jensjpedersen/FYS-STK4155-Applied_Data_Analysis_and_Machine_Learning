import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import sys
import importlib
import franke_data
import ols 
import analysis 
import plot_model
importlib.reload(franke_data)
importlib.reload(ols)
importlib.reload(analysis)
importlib.reload(plot_model)


@dataclass
class PlotData:
    """ Class plots values from dict provided to the initalizer """
    poly_score: dict # Plots values from dict

    poly_deg: np.ndarray = field(init=False)

    def __post_init__(self):
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
        # plt.ylim(0,2)
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

        plt.figure()
        # TODO: Differnet marker colors, 
        style = ['bo', 'ro', 'ko']
        for i, label in enumerate(keys): 
            for p in self.poly_deg:
                betas = self.poly_score[str(p)]['beta'][label]
                x_rep = np.ones(len(betas)) * p
                if p == 1: 
                    # Add label
                    plt.plot(x_rep, betas, style[i], label = label)
                else:
                    plt.plot(x_rep, betas, style[i])

        plt.legend()
        plt.show()


        # XXX: plot horizontal lines
        # # beta = [ self.poly_score[str(j)]['beta'][j] ]
        # for label in keys:
        #     for j in range(n_features): 
        #         betas = []
        #         for i in self.poly_deg:
        #             try: 
        #                 beta_n = self.poly_score[str(i)]['beta'][label][j]
        #             except IndexError: 
        #                 break

        #         betas.append(beta_n)
        #         plt.plot(range(1,len(betas)+1), betas)
        #         plt.show()



        #     print(e)
        #     breakpoint() 


        







        

if __name__ == '__main__':


    np.random.seed(0)
    max_poly_deg = 12
    n_data = 100
    test_size = 0.2
    noise = 1
    data_dim = 2

    # # np.random.seed(3)
    # max_poly_deg = 25
    # n_data = 1000
    # test_size = 0.2
    # noise = 0.1


    f = franke_data.FrankeData(max_poly_deg, n_data, data_dim = data_dim, add_noise = noise, test_size = test_size)
    X_train, X_test, y_train, y_test = f.get_train_test_data()

    score_list = ['mse', 'r2', 'beta']

    method_list = ['ols_own']
    # method_list = ['ols_skl']
    data_list = ['test']

    method_list = ['ols_own', 'ols_skl']
    # method_list = ['ols_skl']
    # data_list = ['test']
    # method_list = ['ols_skl']
    data_list = ['test', 'test']
    


    a = analysis.Analysis(f)
    score = a.calculate_loop(max_poly_deg, score_list, method_list, data_list)

    p = PlotData(score)
    p.plot_beta()
    p.plot_mse()
    p.plot_r2()

    # TODO: implement poly deg parameter in plot model
    pm = plot_model.PlotModel(f)
    pm.plot(method = 'ols_own', data = 'test')
    pm.plot(method = 'ols_skl', data = 'test')
