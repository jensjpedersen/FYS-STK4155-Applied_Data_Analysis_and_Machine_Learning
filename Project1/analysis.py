import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import seaborn as sns
import importlib
import franke_data
import time
import sys
import ipdb
import ols 
import bdb
importlib.reload(franke_data)
importlib.reload(ols)
plt.style.use('fivethirtyeight')
plt.style.use('fivethirtyeight')

def r2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def mse(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def relative_errror(y_data, y_model):
    return abs((y_data-y_model)/y_data)


@dataclass
class Analysis: 
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


    def calculate_loop(self, max_poly_deg:int, score_list = None, method_list = None, data_list = None ):
        """ 
        Parameters:
            max_poly_deg (int) - Scores for all polynomial up to degree = max_poly_deg is calculated
            score_list (list) - List with values to return. score = ['mse', 'r2', 'beta']
            method_list (list) - takes values: 'ols_own' or 'ols_skl'
            data_list (list) - calculate on test and/or training data. values: 'test', 'train'
                               List indices should correspond to method_list
        """
        poly_score = dict() # Return dict with keys correspodning to polynomial degree
        for deg in range(1, max_poly_deg+1):
            poly_score[str(deg)] = self.calculate(deg, score_list, method_list, data_list)

        return poly_score
    
    def calculate(self, deg, score_list = None, method_list = None, data_list = None):
        """ 
        Parameters:
            deg (int) - Polynomail degree
            score_list (list) - List with values to return. score = ['mse', 'r2', 'beta']
            method_list (list) - takes values: 'ols_own' or 'ols_skl'
            data_list (list) - calculate on test and/or training data. values: 'test', 'train'
                               List indices should correspond to method_list
        """

        if score_list == None:
            score_list = []

        if method_list == None:
            method_list = []
        
        if data_list == None:
            data_list = []

        # Slice desing matrix dependent og polynomal degree. 
        l = round(((deg+1)*(deg+2)/2))		# Number of elements in beta

        if l > np.shape(self.X_test)[1]: 
            raise ValueError("""Polynomail degree = {deg} requires {l} features in desing matrix, 
                    has l = {l}. Increse coloumns in desing matrix
                    """)

        # Return dict with scores
        score_dict = dict()
        for score in score_list: 
            if score == 'mse':
                score_dict['mse'] = self.__get_mse(l, method_list, data_list)
            
            if score == 'r2': 
                score_dict['r2'] = self.__get_r2(l, method_list, data_list)

            if score == 'beta':
                score_dict['beta'] = self.__get_beta(l, method_list, data_list)

        return score_dict


    def __get_mse(self, l, method_list, data_list): 
        X_train_deg = self.X_train[:,:l-1]       # Slice matrix -> reduce poly deg
        X_test_deg  = self.X_test[:,:l-1]
        o = ols.OLS(X_train_deg, self.y_train)

        # Return values 
        mse_ols = dict()

        for method, data in zip(method_list, data_list): 

            if method == 'ols_own': 
                o.ols()
                if data == 'train':
                    y_model= o.predict(X_train_deg)
                    mse_ols['ols_own_train'] = mse(self.y_train, y_model)

                elif data == 'test': 
                    y_model = o.predict(X_test_deg)
                    mse_ols['ols_own_test'] = mse(self.y_test, y_model)

            elif method == 'ols_skl': 
                o.skl_ols()
                if data == 'train':
                    y_model = o.predict(X_train_deg)
                    mse_ols['ols_skl_train'] = mse(self.y_train, y_model)

                elif data == 'test': 
                    y_model = o.predict(X_test_deg)
                    mse_ols['ols_skl_test'] = mse(self.y_test, y_model)

        return mse_ols

    def __get_r2(self, l, method_list, data_list): 
        X_train_deg = self.X_train[:,:l-1]       # Slice matrix -> reduce poly deg
        X_test_deg  = self.X_test[:,:l-1]
        o = ols.OLS(X_train_deg, self.y_train)

        # Return values 
        r2_ols = dict()

        for method, data in zip(method_list, data_list): 

            if method == 'ols_own': 
                o.ols()
                if data == 'train':
                    y_model= o.predict(X_train_deg)
                    r2_ols['ols_own_train'] = r2(self.y_train, y_model)

                elif data == 'test': 
                    y_model = o.predict(X_test_deg)
                    r2_ols['ols_own_test'] = r2(self.y_test, y_model)

            elif method == 'ols_skl': 
                o.skl_ols()
                if data == 'train':
                    y_model = o.predict(X_train_deg)
                    r2_ols['ols_skl_train'] = r2(self.y_train, y_model)

                elif data == 'test': 
                    y_model = o.predict(X_test_deg)
                    r2_ols['ols_skl_test'] = r2(self.y_test, y_model)

        return r2_ols

    def __get_beta(self, l, method_list, data_list): 
        X_train_deg = self.X_train[:,:l-1]       # Slice matrix -> reduce poly deg
        X_test_deg  = self.X_test[:,:l-1]
        o = ols.OLS(X_train_deg, self.y_train)

        # Return values 
        beta_ols = dict()

        for method in method_list:

            if method == 'ols_own': 
                beta_ols['ols_own_train'] = o.ols()

            elif method == 'ols_skl': 
                beta_ols['ols_skl_train'] = o.skl_ols()

        return beta_ols


    def __franke_funciton(self, x: np.ndarray, y:np.ndarray):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def plot_model(self, method: str, data: str, data_dim: int): 

        x_train = self.X_train[:, 1] 
        y_train = self.X_train[:, 2]
        
        o = ols.OLS(self.X_train, self.y_train)
        if method == 'ols_own': 
            o.ols()
        elif method == 'ols_skl': 
            o.skl_ols()
        else:
            raise ValueError('Valid methods are: "ols_own", "ols_skl"')


        if data == 'train':
            y_tilde = o.predict(self.X_train)
            x_predict = self.X_train[:, 1]
            y_predict = self.X_train[:, 2]
            z_predict = self.y_train
        elif data == 'test': 
            y_tilde = o.predict(self.X_test)
            x_predict = self.X_test[:, 1]
            y_predict = self.X_test[:, 2]
            z_predict = self.y_test

        # if data_dim == 2:
        #     N = np.sqrt(len(x))
        #     x = x.reshape(N, N)
        #     y = y.reshape(N, N)
        #     z = z.reshape(N, N)
        #     z_train_data = self.y_train
        #     z_model = self.predict(X)



        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if data_dim == 1: 

            # Plot datapoint used for model prediction 
            x_sort_index = np.argsort(x_predict)
            y_sort_index = np.argsort(y_predict)
            assert(np.equal(x_sort_index, y_sort_index).all())
            x = x_predict[x_sort_index]
            y = y_predict[y_sort_index]
            assert(np.equal(x_sort_index, y_sort_index).all())
            z = z_predict[x_sort_index]
            z_model = y_tilde[x_sort_index]

            # ax = fig.add_subplot(111, projection='3d')
            ax.plot(x, y, z, linewidth = 2, label = 'Data points')
            # Plot model 
            ax.plot(x, y, z_model, label = 'Model')

            # Plot Franke function from trianing data, without noise
            x_sort_index = np.argsort(x_train)
            y_sort_index = np.argsort(y_train)
            assert(np.equal(x_sort_index, y_sort_index).all())

            x = x_train[x_sort_index]
            y = y_train[y_sort_index]
            z = self.__franke_funciton(x, y)

            ax.plot(x, y, z, linewidth = 2, label = 'Franke function train data')

            plt.xlabel('x')
            plt.ylabel('y')

            plt.legend()
            plt.show()
            
            sys.exit()

        else:
            raise NotImplementedError


        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # if data_dim == 2:
        #     ax.plot_surface(x, y, z) 
        # elif data_dim == 1:
        #     ax.plot(x, y, z)
        # plt.show()





if __name__ == '__main__': 



    max_poly_deg = 1
    n_data = 200000
    n_data = 20000
    test_size = 0.2

    f = franke_data.FrankeData(max_poly_deg, n_data, data_dim = 1, add_noise = 0)
    X_train, X_test, y_train, y_test = f.get_train_test_data(test_size = test_size) # XXX pass to function call
    
    # f.print_design_matrix()


    a = Analysis(X_train, X_test, y_train, y_test)
    a.plot_model(method = 'ols_own', data = 'train', data_dim = 1)


    method = ['ols_own', 'ols_own', 'ols_skl']
    data = ['train', 'test', 'test']
    score = ['r2', 'mse']
    tic = time.perf_counter()
    s = a.calculate(max_poly_deg, score, method, data)

    ps = a.calculate_loop(max_poly_deg, score, method, data)

