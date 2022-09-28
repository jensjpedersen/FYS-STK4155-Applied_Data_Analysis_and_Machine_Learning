import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import sys
import sklearn as skl
import importlib
import franke_data
import plot_data
import ols 
import lasso_regression
import ridge_regression
import analysis 
importlib.reload(franke_data)
importlib.reload(ols)
importlib.reload(lasso_regression)
importlib.reload(ridge_regression)
importlib.reload(analysis)
importlib.reload(plot_data)


@dataclass(frozen=True) 
class ResamplingAnalysis: 
    franke_object: franke_data.FrankeData


    def boots_lamb_loop(self, regression_methods: list, n_resamples:int, resample_dataset, lambda_list: list):
        keys = ['lamb:'+str(l) for l in lambda_list]
        boots_lamb_scores = {key: {} for key in keys}

        y_test = self.franke_object.get_y_test()
        max_poly_deg = self.franke_object.n
        
        for lamb in lambda_list: 
            for deg in range(1, max_poly_deg + 1): 
                boots_lamb_scores[f'lamb:{str(lamb)}'][str(deg)] = {'mse': {}}
                for method in regression_methods: 
                    re = Resampling(self.franke_object, poly_deg = deg, lamb = lamb)
                    y_boots_pred = re.bootstrap(n_resamples, method, resample_dataset)
                    rs = ResamplingScores(y_test, y_boots_pred)
                    boots_lamb_scores[f'lamb:{str(lamb)}'][str(deg)]['mse'][method] = rs.mse()

        return boots_lamb_scores

        




    def kfold_loop(self, regression_methods: list, n_splits: int, dataset: str, lamb: float): 
        """
        Parameters:
            regression_methods (list): list of regression methods to predict scores from
            n_splits (int): Split datset in n splits where one split is used for testing, 
                            and the others for training
            dataset (str): Chose dataset to split. The dataset splits is used for both 
                           testing and training. """

        kfold_scores = dict()

        max_poly_deg = self.franke_object.n

        y_test = self.franke_object.get_y_test()

        # # Get datset 
        # get_func_name = f'get_{dataset}'
        # get_data = getattr(self.franke_object, get_func_name)
        # X_data, y_data = get_data(deg = self.poly_deg)
        # XXX: not here need deg
        
        for deg in range(1, max_poly_deg + 1): 
            # kfold_scores[str(deg)] = {'mse': {}, 'bias': {}, 'variance': {}}
            kfold_scores[str(deg)] = {'mse': {}}

            re = Resampling(self.franke_object, poly_deg = deg, lamb=lamb)

            for method in regression_methods:
                # y_kfold_test, y_kfold_pred = re.kfold_own(n_splits=n_splits, dataset=dataset, regression_method=method)
                # XXX: own method not working
                # mse = re.kfold_own(n_splits=n_splits, dataset=dataset, regression_method=method)
                # kfold_scores[str(deg)]['mse'][str(method)] = mse

                kfold_scores[str(deg)]['mse'][str(method)] = re.kfold_skl(n_splits, method, dataset) # XXX: test

        return kfold_scores



    def bootstrap_loop(self, regression_methods: list, n_resamples:int, resample_dataset, lamb: float):
        """ Loops through polynoamil degree and caculates scores (mse, bias, variance) 
        Parameters:
            regression_methods (list): list of regression methods to predict scores from
            n_resamples: 
            resample_dataset (str): Chosses dataset to draw samples from. 
                                    Values: 'test', 'train', 'validation' (not implemented). 
                                    Satistics is always predicted on the test data defined in franke_object. 
        Returns:
            resampling_scores: dict with scores """

        y_test = self.franke_object.get_y_test()
        max_poly_deg = self.franke_object.n

        resampling_scores = dict()
        # resampling_scores = {'mse': {}, 'bias': {}, 'variance': {}}
        
        for deg in range(1, max_poly_deg + 1): 
            resampling_scores[str(deg)] = {'mse': {}, 'bias': {}, 'variance': {}}

            # l = int(((deg+1)*(deg+2)/2))		# Number of elements in beta
            # _X_train = self.X_train[:, :l] 
            # _X_test = self.X_test[:, :l]

            
            
            # bias variance tradeoff make sence on train data
            # XXX:chenaged args to f
            # slice with respect to deg
            # r = Resampling(_X_train, _X_test, self.y_train)

            for method in regression_methods: 
                re = Resampling(self.franke_object, poly_deg = deg, lamb = lamb)
                y_boots_pred = re.bootstrap(n_resamples, method, resample_dataset)
                # XXX: arg poly_deg
                rs = ResamplingScores(y_test, y_boots_pred)
                resampling_scores[str(deg)]['mse'][str(method)] = rs.mse()
                resampling_scores[str(deg)]['bias'][str(method)] = rs.bias()
                resampling_scores[str(deg)]['variance'][str(method)] = rs.variance() 

        return resampling_scores



@dataclass(frozen=True)
class ResamplingScores: 
    y_test: np.ndarray
    y_pred: np.ndarray
    # score_dict: dict = field(init=False)

    def __post_init__(self):
        assert(np.shape(self.y_test)[0] == np.shape(self.y_pred)[0])
        if len(np.shape(self.y_pred)) != 2:
            raise ValueError('y_pred should be resampled in n samples')

        if len(np.shape(self.y_test)) == 1: 
            y_test = self.y_test.reshape(np.shape(self.y_pred)[0], 1) # reshape array for subtraction to work 
            object.__setattr__(self, 'y_test', y_test)

    def bias(self): 
        # y_test = self.y_test.reshape(np.shape(self.y_pred)[0], 1) # reshape array for subtraction to work 
        bias = np.mean( (self.y_test - np.mean(self.y_pred, axis=1, keepdims=True))**2 )
        return bias

    def variance(self): 
        # y_test = self.y_test.reshape(np.shape(self.y_pred)[0], 1) # reshape array for subtraction to work 
        variance = np.mean( np.var(self.y_pred, axis=1, keepdims=True) )
        return variance

    def mse(self): 
        # y_test = self.y_test.reshape(np.shape(self.y_pred)[0], 1) # reshape array for subtraction to work 
        mse = np.mean( np.mean((self.y_test - self.y_pred)**2, axis=1, keepdims=True) )
        return mse


@dataclass(frozen=True)
class Resampling: 
    """Parameters:
        franke_object
        poly_deg (int): Desing matrix is sliced with respect to polynomail degree
        lamb (double): hyper parameter
    """
    franke_object: franke_data.FrankeData
    poly_deg: int 
    lamb: float


    def kfold_own(self, n_splits: int, regression_method:str, dataset: str): 
        """
        Parameters:
            Dataset (str) - data used for kfold splitting. datset is devided in n_splits,
                            where one fold is used for testing and the others for training. 
        Returns:
            y_test: shape (n_data/n_splits, n_splits) 
            y_pred: Matrix with predicted values of size:(n_data/n_splits, n_splits)
        """
        mse_kfold_deg = [] # XXX: rm 

        # Get desing matrix
        get_func_name = f'get_{dataset}'
        get_data = getattr(self.franke_object, get_func_name)
        X_data, y_data = get_data(deg = self.poly_deg)

        n_data = len(y_data)
        n_test_data = int(np.ceil(len(y_data)/n_splits))
        y_pred_kfold = np.zeros((n_test_data, n_splits))
        y_test_kfold = np.zeros((n_test_data, n_splits))
        # y_pred_kfold_test = np.zeros((n_test_data, n_splits))
        idx = np.arange(n_data)
        shuffle_idx = np.ones(n_data) 
        for i in range(n_splits): 
            old_shuf_idx = shuffle_idx
            shuffle_idx = np.random.choice(idx, size=n_data, replace=False)
            assert((old_shuf_idx != shuffle_idx).any())
            test_idx = list(shuffle_idx[:n_test_data])
            train_idx = list(shuffle_idx[n_test_data:])

            X_kfold_train = X_data[train_idx,:]
            y_kfold_train = y_data[train_idx]
            X_kfold_test = X_data[test_idx,:]
            y_kfold_test = y_data[test_idx]

            # y_test_kfold[:,i] = y_kfold_test
            # y_pred_kfold[:,i] = self.regression_method(regression_method, X_kfold_train, y_kfold_train, X_kfold_test)

            # # XXX test  Fix
            o = ols.OLS(X_kfold_train, y_kfold_train)
            o.ols()
            y_kfold_pred = o.predict(X_kfold_test)

            # mse = np.mean( np.mean((y_kfold_test[:,np.newaxis] - y_kfold_pred[:,np.newaxis])**2, axis=1, keepdims=True) )
            mse = np.mean((y_kfold_test[:,np.newaxis] - y_kfold_pred[:,np.newaxis])**2) 
            mse_kfold_deg.append(mse)

            # return mse

        return np.mean(mse_kfold_deg)



        # print(self.poly_deg)
        # print(np.mean(mse_test))
        return y_test_kfold, y_pred_kfold

    def kfold_skl(self, n_splits: int, regression_method: str, dataset: str):
        """ Parameters:
                Dataset (str) - data used for kfold splitting. datset is devided in n_splits,
                                where one fold is used for testing and the others for training. 
        """

        # Get desing matrix
        get_func_name = f'get_{dataset}'
        get_data = getattr(self.franke_object, get_func_name)
        X_data, y_data = get_data(deg = self.poly_deg)

        kfold = skl.model_selection.KFold(n_splits = n_splits)

        n_test_data = int(np.ceil(len(y_data)/n_splits))
        y_pred_kfold_test = np.zeros((n_test_data, n_splits))
        i = 0
        mse_kfold_deg = []
        for train_idx, test_idx in kfold.split(y_data): 
            # XXX: differnet test sizes
            # raise IndexError('len(test_idx) differ')
            X_kfold_train = X_data[train_idx,:]
            y_kfold_train = y_data[train_idx]
            X_kfold_test = X_data[test_idx,:]
            y_kfold_test = y_data[test_idx]
            # y_pred_kfold_test[:,i] = self.regression_method(regression_method, X_kfold_train, y_kfold_train, X_kfold_test)

            # o = ols.OLS(X_kfold_train, y_kfold_train)
            # o.ols()
            # y_kfold_pred = o.predict(X_kfold_test)

            y_kfold_pred = self.regression_method(regression_method, X_kfold_train, y_kfold_train, X_kfold_test, lamb = self.lamb)

            mse = np.mean( np.mean((y_kfold_test[:,np.newaxis] - y_kfold_pred[:,np.newaxis])**2, axis=1, keepdims=True) )
            mse_kfold_deg.append(mse)

        # return mse
            i += 1

        return np.mean(mse_kfold_deg)

    def bootstrap(self, n_resamples: int, method: str, resample_dataset:str) -> np.ndarray:
        """
        Parameters:
            n_resamples: Number of resmaples
            method (str): Regression method that is used on test data.
                          Values: 'ols_skl', 'ols_own'

        Returns:
            y-pred: Matrix with predicted values of size:(n_test_data, n_resamples)
        """
        # Define trianing and test data
        X_test = self.franke_object.get_X_test(deg=self.poly_deg)

        get_func_name = f'get_{resample_dataset}'
        get_data = getattr(self.franke_object, get_func_name)
        X_resample, y_resample = get_data(deg=self.poly_deg) # Datset to resample

        y_pred = np.zeros((np.shape(X_test)[0], n_resamples))

        for i in range(n_resamples):
            X_, y_ = skl.utils.resample(X_resample, y_resample)
            y_pred[:,i] = self.regression_method(method, X_, y_, X_test, lamb = self.lamb)

        return y_pred

    def regression_method(self, method, X_train, y_train, X_test, lamb):
        """ 
        Parameters:
            method (str): regression_method = ols_own, ols_skl, ridge_own, ridge_skl, lasso_skl 
        """
        assert(len(X_train) == len(y_train))
        assert(len(X_train) > len(X_test))
        X_ = X_train
        y_ = y_train

        if method == 'ols_skl': 
            o = ols.OLS(X_, y_)
            o.skl_ols()
            y_pred = o.predict(X_test)

        elif method == 'ols_own': 
            o = ols.OLS(X_, y_)
            o.ols()
            y_pred = o.predict(X_test)
        elif method == 'ridge_own': 
            r = ridge_regression.RidgeRegression(X_, y_, lamb)
            r.ridge_own()
            y_pred = r.predict(X_test)
        elif method == 'ridge_skl':
            r = ridge_regression.RidgeRegression(X_, y_, lamb)
            r.ridge_skl()
            y_pred = r.predict(X_test)
        elif method == 'lasso_skl':
            l = lasso_regression.LassoRegression(X_, y_, lamb)
            l.lasso_skl()
            y_pred = l.predict(X_test)
        else:
            raise ValueError('Valide methods are: ols_skl, ols_own')

        return y_pred



def plot_bias_variance_tradeoff(scores, score_list: None): 
    """ Plots scores from scores dict in same plot"""
    # Get dict info
    poly_deg = np.array([ int(i) for i in scores ])
    score_names = list(scores['1'].keys())
    method_names = list(scores['1'][score_names[0]].keys())

    if score_list == None:
        raise ValueError("Spesify list with scors: eg. score_list = ['mse', 'bias', 'variance']")
    # Plot mse, bias and variance.  
    for method in method_names: 
        for score in score_list: 
            vals = np.array([ scores[str(deg)][score][method] for deg in poly_deg ])
            plt.plot(poly_deg, vals, label = f'{score} from {method}')


    plt.legend()
    plt.xlabel('Polynomial degree')
    plt.show()
            



if __name__ == '__main__': 

    # n = 12   # Poly deg
    # n = 8   # Poly deg
    # N = 100 # dataset size
    # noise = 1


    # np.random.seed(0)
    max_poly_deg = 12
    # max_poly_deg = 8
    n_data = 50
    # n_data = 50
    test_size = 0.2
    noise = 0.5
    data_dim = 2

    n_resamples = 50


    f = franke_data.FrankeData(max_poly_deg, n_data, data_dim = data_dim, add_noise = noise, test_size = test_size, set_seed=True)

    X_train, X_test, y_train, y_test = f.get_train_test_data()


    # max_poly_deg = n
    methods = ['ols_own']
    methods = ['ols_skl']
    methods = ['ols_own', 'ols_skl']

    # XXX conitnue plot ordinary scores 
    # a = analysis.Analysis(f)
    # score = a.calculate_loop(max_poly_deg, ['mse'], ['ols_skl', 'ols_own'], ['test', 'test'])
    # p = plot_data.PlotData(score)
    # p.plot_mse()

    ra = ResamplingAnalysis(f)
    # XXX: does not look good
    # Debug: plot mse for each y_pred_kfold, y_test_kfold




    # =============== Kfold ===============
    # regression_methods = ['ols_own', 'ols_skl', 'ridge_own', 'ridge_skl', 'lasso_skl']
    # regression_methods = ['ridge_own', 'ols_own']
    # kfold_scores = ra.kfold_loop(regression_methods = regression_methods, n_splits=5, dataset='train', lamb = 0.001)
    # score_list = ['mse']
    # plot_bias_variance_tradeoff(kfold_scores, score_list=score_list)

    # =============== Boots ===============
    # regression_methods = ['ols_own', 'ols_skl', 'ridge_own', 'ridge_skl', 'lasso_skl']
    # regression_methods = ['lasso_skl', 'ols_own']
    # score_list = ['mse']
    # boots_scores = ra.bootstrap_loop(regression_methods = regression_methods, n_resamples = 20, resample_dataset='train', lamb = 0.0001)
    # plot_bias_variance_tradeoff(boots_scores, score_list=score_list)

    # =============== Boots lambdas ===============
    lambda_list = np.logspace(-3, 3, 7)

    regression_methods = ['ridge_own', 'ridge_skl']
    lamb_scores = ra.boots_lamb_loop(regression_methods, n_resamples = 20, resample_dataset='train', lambda_list = lambda_list)
    
    




