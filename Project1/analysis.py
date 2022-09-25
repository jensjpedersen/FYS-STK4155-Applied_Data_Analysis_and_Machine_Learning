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


@dataclass(frozen=True)
class Analysis: 
    franke_object: franke_data.FrankeData
    # X_train: np.ndarray
    # X_test: np.ndarray
    # y_train: np.ndarray
    # y_test: np.ndarray


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
        l = int(((deg+1)*(deg+2)/2))		# Number of elements in beta

        if l > np.shape(self.franke_object.get_X_test())[1]: 
            raise ValueError("""Polynomail degree = {deg} requires {l} features in desing matrix, 
                    has l = {l}. Increse coloumns in desing matrix
                    """)

        # Return dict with scores
        score_dict = dict()
        for score in score_list: 
            if score == 'mse':
                score_dict['mse'] = self.__get_mse(deg, method_list, data_list)
            
            if score == 'r2': 
                score_dict['r2'] = self.__get_r2(deg, method_list, data_list)

            if score == 'beta':
                score_dict['beta'] = self.__get_beta(deg, method_list)

        return score_dict


    def __get_mse(self, deg, method_list, data_list): 
        X_train_deg = self.franke_object.get_X_train(deg)
        X_test_deg = self.franke_object.get_X_test(deg)
        y_train = self.franke_object.get_y_train()
        y_test = self.franke_object.get_y_test()
        o = ols.OLS(X_train_deg, y_train)

        # Return values 
        mse_ols = dict()

        for method, data in zip(method_list, data_list): 

            if method == 'ols_own': 
                o.ols()
                if data == 'train':
                    y_model= o.predict(X_train_deg)
                    mse_ols['ols_own_train'] = mse(y_train, y_model)

                elif data == 'test': 
                    y_model = o.predict(X_test_deg)
                    mse_ols['ols_own_test'] = mse(y_test, y_model)

            elif method == 'ols_skl': 
                o.skl_ols()
                if data == 'train':
                    y_model = o.predict(X_train_deg)
                    mse_ols['ols_skl_train'] = mse(y_train, y_model)

                elif data == 'test': 
                    y_model = o.predict(X_test_deg)
                    mse_ols['ols_skl_test'] = mse(y_test, y_model)

        return mse_ols

    def __get_r2(self, deg, method_list, data_list): 
        X_train_deg = self.franke_object.get_X_train(deg)
        X_test_deg = self.franke_object.get_X_test(deg)
        y_train = self.franke_object.get_y_train()
        y_test = self.franke_object.get_y_test()
        o = ols.OLS(X_train_deg, y_train)

        # Return values 
        r2_ols = dict()

        for method, data in zip(method_list, data_list): 

            if method == 'ols_own': 
                o.ols()
                if data == 'train':
                    y_model= o.predict(X_train_deg)
                    r2_ols['ols_own_train'] = r2(y_train, y_model)

                elif data == 'test': 
                    y_model = o.predict(X_test_deg)
                    r2_ols['ols_own_test'] = r2(y_test, y_model)

            elif method == 'ols_skl': 
                o.skl_ols()
                if data == 'train':
                    y_model = o.predict(X_train_deg)
                    r2_ols['ols_skl_train'] = r2(y_train, y_model)

                elif data == 'test': 
                    y_model = o.predict(X_test_deg)
                    r2_ols['ols_skl_test'] = r2(y_test, y_model)

        return r2_ols

    def __get_beta(self, deg, method_list): 
        X_train_deg = self.franke_object.get_X_train(deg)
        y_train = self.franke_object.get_y_train()
        o = ols.OLS(X_train_deg, y_train)

        # Return values 
        beta_ols = dict()

        for method in method_list:

            if method == 'ols_own': 
                beta_ols['ols_own_train'] = o.ols()

            elif method == 'ols_skl': 
                beta_ols['ols_skl_train'] = o.skl_ols()

        return beta_ols


if __name__ == '__main__': 




    np.random.seed(0)
    max_poly_deg = 3
    n_data = 1000
    # n_data = 2000000
    test_size = 0.2
    noise = 0

    f = franke_data.FrankeData(max_poly_deg, n_data, data_dim = 1, add_noise = noise, test_size = test_size)
    X_train, X_test, y_train, y_test = f.get_train_test_data() # XXX pass to function call

    # XXX: change takes franke object
    a = Analysis(f) 

    method = ['ols_own', 'ols_skl']
    data = ['test', 'test']
    score = ['r2', 'mse']
    # score = ['beta']
    s = a.calculate(max_poly_deg, score, method, data)
    # print(s)

    ps = a.calculate_loop(max_poly_deg, score, method, data)

