import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import importlib
import franke_data
import time
import sys
import ipdb
import ols 
import bdb
importlib.reload(franke_data)
importlib.reload(ols)

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






if __name__ == '__main__': 



    max_poly_deg = 6
    n_data = 200000
    test_size = 0.2

    f = franke_data.FrankeData(max_poly_deg, n_data, data_dim = 1, add_noise = 1)
    X_train, X_test, y_train, y_test = f.get_train_test_data(test_size = test_size) # XXX pass to function call
    
    # f.print_design_matrix()


    a = Analysis(X_train, X_test, y_train, y_test)


    method = ['ols_own', 'ols_own', 'ols_skl']
    data = ['train', 'test', 'test']
    score = ['r2', 'mse']
    tic = time.perf_counter()
    s = a.calculate(max_poly_deg, score, method, data)

    ps = a.calculate_loop(max_poly_deg, score, method, data)

