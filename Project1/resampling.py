import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import sys
import sklearn as skl
import importlib
import warnings
import seaborn as sns
import franke_data
import plot_data
import ols 
import logging
import lasso_regression
import ridge_regression
import analysis 
import terrain_data
importlib.reload(franke_data)
importlib.reload(ols)
importlib.reload(lasso_regression)
importlib.reload(ridge_regression)
importlib.reload(analysis)
importlib.reload(plot_data)




numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
logging.basicConfig(filename='debug.log', encoding='utf-8', level=logging.DEBUG, force=True)

@dataclass(frozen=True) 
class ResamplingAnalysis: 
    data_object: franke_data.FrankeData | terrain_data.TerrainData


    def boots_lamb_loop(self, regression_methods: list, n_resamples:int, resample_dataset, predict_dataset: str, lambda_list: list):
        keys = ['lamb:'+str(l) for l in lambda_list]
        boots_lamb_scores = {key: {} for key in keys}

        y_test = self.data_object.get_y_test()
        max_poly_deg = self.data_object.n
        
        for lamb in lambda_list: 
            for deg in range(1, max_poly_deg + 1): 
                boots_lamb_scores[f'lamb:{str(lamb)}'][str(deg)] = {'mse': {}}
                for method in regression_methods: 
                    re = Resampling(self.data_object, poly_deg = deg, lamb = lamb)
                    y_boots_pred = re.bootstrap(n_resamples, method, resample_dataset, predict_dataset)
                    rs = ResamplingScores(y_test, y_boots_pred)
                    boots_lamb_scores[f'lamb:{str(lamb)}'][str(deg)]['mse'][method] = rs.mse()

        return boots_lamb_scores

    def kfold_lamb_loop(self, regression_methods: list, n_splits:int, dataset: str, lambda_list: list):
        keys = ['lamb:'+str(l) for l in lambda_list]
        kfold_lamb_scores = {key: {} for key in keys}

        y_test = self.data_object.get_y_test()
        max_poly_deg = self.data_object.n
        
        for lamb in lambda_list: 
            for deg in range(1, max_poly_deg + 1): 
                kfold_lamb_scores[f'lamb:{str(lamb)}'][str(deg)] = {'mse': {}}
                for method in regression_methods: 
                    re = Resampling(self.data_object, poly_deg = deg, lamb = lamb)
                    mse = re.kfold_skl(n_splits, method, dataset) # FIXME: skl kfold
                    kfold_lamb_scores[f'lamb:{str(lamb)}'][str(deg)]['mse'][method] = mse

        return kfold_lamb_scores

        




    def kfold_loop(self, regression_methods: list, n_splits: int, dataset: str, lamb: float): 
        """
        Parameters:
            regression_methods (list): list of regression methods to predict scores from
            n_splits (int): Split datset in n splits where one split is used for testing, 
                            and the others for training
            dataset (str): Chose dataset to split. The dataset splits is used for both 
                           testing and training. """

        kfold_scores = dict()

        max_poly_deg = self.data_object.n

        # y_test = self.data_object.get_y_test()

        # # Get datset 
        # get_func_name = f'get_{dataset}'
        # get_data = getattr(self.data_object, get_func_name)
        # X_data, y_data = get_data(deg = self.poly_deg)
        # XXX: not here need deg
        
        for deg in range(1, max_poly_deg + 1): 
            # kfold_scores[str(deg)] = {'mse': {}, 'bias': {}, 'variance': {}}
            kfold_scores[str(deg)] = {'mse': {}, 'mse_skl': {}}

            re = Resampling(self.data_object, poly_deg = deg, lamb=lamb)

            for method in regression_methods:
                # Own kfold
                # mse = re.kfold_own(n_splits=n_splits, dataset=dataset, regression_method=method)
                # kfold_scores[str(deg)]['mse'][str(method)] = mse

                # Skl kfold
                kfold_scores[str(deg)]['mse'][str(method)] = re.kfold_skl(n_splits, method, dataset) 



        return kfold_scores



    def bootstrap_loop(self, regression_methods: list, n_resamples:int, resample_dataset, predict_dataset, lamb: float):
        """ Loops through polynoamil degree and caculates scores (mse, bias, variance) 
        Parameters:
            regression_methods (list): list of regression methods to predict scores from
            n_resamples: 
            resample_dataset (str): Chosses dataset to draw samples from. 
                                    Values: 'test', 'train', 'validation' (not implemented). 
                                    Satistics is always predicted on the test data defined in data_object. 
        Returns:
            resampling_scores: dict with scores """
        logging.info(self.bootstrap_loop)

        # XXX: add parameter predict_dataset
        # y_test = self.data_object.get_y_test()
        max_poly_deg = self.data_object.n

        get_test_data = f'get_y_{predict_dataset}'
        get_test_data = getattr(self.data_object, get_test_data)
        y_test = get_test_data()

        resampling_scores = dict()
        # resampling_scores = {'mse': {}, 'bias': {}, 'variance': {}}
        
        for deg in range(1, max_poly_deg + 1): 
            resampling_scores[str(deg)] = {'mse': {}, 'bias': {}, 'variance': {}, 'r2': {}}

            # l = int(((deg+1)*(deg+2)/2))		# Number of elements in beta
            # _X_train = self.X_train[:, :l] 
            # _X_test = self.X_test[:, :l]

            
            
            # bias variance tradeoff make sence on train data
            # XXX:chenaged args to f
            # slice with respect to deg
            # r = Resampling(_X_train, _X_test, self.y_train)

            for method in regression_methods: 
                re = Resampling(self.data_object, poly_deg = deg, lamb = lamb)
                # TODO: pred on test and train
                
                y_boots_pred = re.bootstrap(n_resamples, method, resample_dataset, predict_dataset)
                # XXX: arg poly_deg
                rs = ResamplingScores(y_test, y_boots_pred)
                # resampling_scores[str(deg)]['mse'][str(method)] = rs.mse()
                # resampling_scores[str(deg)]['bias'][str(method)] = rs.bias()
                # resampling_scores[str(deg)]['variance'][str(method)] = rs.variance() 

                resampling_scores[str(deg)]['mse'][f'{method}_{predict_dataset}'] = rs.mse()
                resampling_scores[str(deg)]['bias'][f'{method}_{predict_dataset}'] = rs.bias()
                resampling_scores[str(deg)]['variance'][f'{method}_{predict_dataset}'] = rs.variance() 
                resampling_scores[str(deg)]['r2'][f'{method}_{predict_dataset}'] = rs.r2() 

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

    def r2(self):
        r2 = 1 - np.sum((self.y_test - self.y_pred) ** 2) / np.sum((self.y_test - np.mean(self.y_pred)) ** 2)
        return r2


@dataclass(frozen=True)
class Resampling: 
    """Parameters:
        data_object
        poly_deg (int): Desing matrix is sliced with respect to polynomail degree
        lamb (double): hyper parameter
    """
    data_object: franke_data.FrankeData | terrain_data.TerrainData
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
        get_data = getattr(self.data_object, get_func_name)
        X_data, y_data = get_data(deg = self.poly_deg)

        n_data = len(y_data)
        n_test_data = int(np.ceil(len(y_data)/n_splits))
        # y_pred_kfold = np.zeros((n_test_data, n_splits))
        # y_test_kfold = np.zeros((n_test_data, n_splits))
        # y_pred_kfold_test = np.zeros((n_test_data, n_splits))
        idx_shuffled = np.arange(n_data)
        np.random.shuffle(idx_shuffled)


        mse_kfold_deg = []
        for i in range(n_splits): 
            test_idx = idx_shuffled[n_test_data*i:n_test_data*(i+1)]
            train_idx = np.delete(idx_shuffled, np.s_[n_test_data*i:n_test_data*(i+1)])

            X_kfold_train = X_data[train_idx,:]
            y_kfold_train = y_data[train_idx]
            X_kfold_test = X_data[test_idx,:]
            y_kfold_test = y_data[test_idx]

            # y_test_kfold[:,i] = y_kfold_test
            # y_pred_kfold[:,i] = self.regression_method(regression_method, X_kfold_train, y_kfold_train, X_kfold_test)

            # # XXX test  Fix

    # def regression_method(self, method, X_train, y_train, X_test, lamb):
            

            y_kfold_pred = self.regression_method(regression_method, X_kfold_train, y_kfold_train, X_kfold_test, lamb = self.lamb)
            mse = np.mean((y_kfold_test[:,np.newaxis] - y_kfold_pred[:,np.newaxis])**2) 
            mse_kfold_deg.append(mse)

        return np.mean(mse_kfold_deg)

        # return np.mean(mse_kfold_deg)



        # print(self.poly_deg)
        # print(np.mean(mse_test))
        # return y_test_kfold, y_pred_kfold

    def kfold_skl(self, n_splits: int, regression_method: str, dataset: str):
        """ 
        Parameters:
            Dataset (str) - data used for kfold splitting. datset is devided in n_splits,
                            where one fold is used for testing and the others for training. 

        Returns:
             np.mean(mse_kfold_deg)
        """

        # Get desing matrix
        get_func_name = f'get_{dataset}'
        get_data = getattr(self.data_object, get_func_name)
        X_data, y_data = get_data(deg = self.poly_deg)

        kfold = skl.model_selection.KFold(n_splits = n_splits)

        # n_test_data = int(np.ceil(len(y_data)/n_splits))
        # y_pred_kfold_test = np.zeros((n_test_data, n_splits))
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

    def bootstrap(self, n_resamples: int, method: str, resample_dataset:str, predict_dataset: str) -> np.ndarray:
        """
        Parameters:
            n_resamples: Number of resmaples
            method (str): Regression method that is used on test data.
                          Values: 'ols_skl', 'ols_own'

        Returns:
            y-pred: Matrix with predicted values of size:(n_test_data, n_resamples)
        """


        logging.info(f"""method:Resampling.bootstrap
        Parameters:
            n_resamples: {n_resamples}
            method: {method}
            resample_dataset: {resample_dataset}
            predict_dataset: {predict_dataset}
        """)
        # Get test data
        # X_test = self.data_object.get_X_test(deg=self.poly_deg)
        get_test_data = f'get_X_{predict_dataset}'
        get_test_data = getattr(self.data_object, get_test_data)
        X_test = get_test_data(deg = self.poly_deg)

        logging.info(f"""get_test_data: {get_test_data} 
        poly_deg = {self.poly_deg} 
        X_test shape: {np.shape(X_test)}""")

        # Get training data
        get_func_name = f'get_{resample_dataset}'
        get_data = getattr(self.data_object, get_func_name)
        X_resample, y_resample = get_data(deg=self.poly_deg) # Datset to resample

        logging.info(f"""get_train_data: {get_data} 
        poly_deg = {self.poly_deg}
        X_resample shape: {np.shape(X_resample)}
        y_resample shape: {np.shape(y_resample)}""")

        y_pred = np.zeros((np.shape(X_test)[0], n_resamples))

        for i in range(n_resamples):
            X_, y_ = skl.utils.resample(X_resample, y_resample)
            assert(np.shape(y_) == np.shape(y_resample))
            y_pred[:,i] = self.regression_method(method, X_, y_, X_test, lamb = self.lamb)

        logging.info(f' y_pred shape: {np.shape(y_pred)}')
        return y_pred

    def regression_method(self, method, X_train, y_train, X_test, lamb):
        """ 
        Parameters:
            method (str): regression_method = ols_own, ols_skl, ridge_own, ridge_skl, lasso_skl 
        """
        assert(len(X_train) == len(y_train))
        assert(len(X_train) >= len(X_test))
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



def plot_bias_variance_tradeoff(scores, score_list: None, title: str = None, filename: str = None): 
    """ Plots scores from scores dict in same plot"""
    # Get dict info
    poly_deg = np.array([ int(i) for i in scores ])
    score_names = list(scores['1'].keys())
    method_names = list(scores['1'][score_names[0]].keys())

    if score_list == None:
        raise ValueError("Spesify list with scors: eg. score_list = ['mse', 'bias', 'variance']")
    # Plot mse, bias and variance.  

    plt.figure(figsize = (12,8))
    for method in method_names: 
        for score in score_list: 
            vals = np.array([ scores[str(deg)][score][method] for deg in poly_deg ])
            plt.plot(poly_deg, vals, label = f'{score} from {method}')


    plt.title(title)
    plt.legend()
    plt.xlabel('Polynomial degree')
    plt.show()

def plot_heatmap(scores: dict, score_list: list, reg_method: str, title: str = None, deg_min = 1): 
    """ 
    Parameters:
        reg_method_list: [ridge_own, ols_own, ...] - Must correspond to key in scores dict
        score_list: [mse, bias, ... ] - Must correspond to key in scores dict
        deg_min : mminimum polynomail degree to include in heat map
    """

    # get lambda
    lambdas_keys = list(scores.keys())
    lambdas = [ float(lamb.replace('lamb:','')) for lamb in lambdas_keys ]
    poly_deg_keys = list(scores[lambdas_keys[0]])
    poly_deg = [ int(deg) for deg in poly_deg_keys ]

    for score_ in score_list:
        M = np.zeros((len(lambdas), len(poly_deg)))
        for j, lamb in enumerate(lambdas_keys): 
            for i, deg in enumerate(poly_deg_keys):
                M[j,i] = scores[lamb][deg][score_][reg_method]


        plt.figure(figsize=(25,8))
        M_min = np.min(M)
        M_max = np.max(M)
        vmin = M_min
        vmax = M_min + (M_max - M_min)/50
        # vmax = 0.03
        sns.heatmap(M[:, deg_min-1:], annot=True, fmt='.1f',
                vmax = vmax, 
                cbar_kws={'label': score_.upper()}, 
                xticklabels = [str(deg) for deg in poly_deg[deg_min-1:]],
                yticklabels=[str(lamb) for lamb in lambdas]) 

        if title != None: 
            plt.title(title)

        plt.xlabel('Polynomial degree')
        plt.ylabel(r'$\lambda$')
        plt.show()



if __name__ == '__main__': 

    # logging.basicConfig(filename='debug.log')

    # n = 12   # Poly deg
    # n = 8   # Poly deg
    # N = 100 # dataset size
    # noise = 1

    np.random.seed(1)

    # np.random.seed(0)
    max_poly_deg = 12
    # max_poly_deg = 8
    n_data = 20
    # n_data = 50
    test_size = 0.2
    noise = 0.2
    # noise = 0
    data_dim = 2

    f = franke_data.FrankeData(max_poly_deg, n_data, data_dim = data_dim, add_noise = noise, test_size = test_size, set_seed=True)
    ra = ResamplingAnalysis(f)

    # =============== Resampling parameters ===============
    n_resamples = 100
    lambda_list = np.logspace(-6, 1, 8)
    n_splits = 10

    # # =============== Kfold ===============
    # regression_methods = ['ols_own', 'ols_skl', 'ridge_own', 'ridge_skl', 'lasso_skl']
    # regression_methods = ['ridge_own', 'ols_own', 'ridge_skl']
    # regression_methods = ['ols_own']
    # n_splits = 10
    # kfold_scores = ra.kfold_loop(regression_methods = regression_methods, n_splits=n_splits, dataset='data', lamb = 0.001)
    # # score_list = ['mse', 'mse_skl']
    # score_list = ['mse']
    # # plot_bias_variance_tradeoff(kfold_scores, score_list=score_list, title = f'kfold cross validation with n = {n_splits} splits')
    # plot_bias_variance_tradeoff(kfold_scores, score_list=score_list, title = f'Kfold cross validation with n = {n_splits} splits')

    # # =============== Boots ===============
    # regression_methods = ['ols_own', 'ols_skl', 'ridge_own', 'ridge_skl', 'lasso_skl']
    # regression_methods = ['ols_own', 'ols_skl']
    # score_list = ['mse', 'bias', 'variance']
    # n_resamples = 100
    # boots_scores = ra.bootstrap_loop(regression_methods = regression_methods,
    #         n_resamples = n_resamples, resample_dataset='train', lamb = 0.0001, predict_dataset='test')
    # plot_bias_variance_tradeoff(boots_scores, score_list=score_list, 
    #         title = f'Bootstrap with n resamples = {n_resamples} and n datapoints = {n_data*n_data}')



    # # TODO: add possibility to predict on trian data
    # # =============== Boots lambdas heatmap ===============
    # regression_methods = ['ridge_own']
    # lamb_scores = ra.boots_lamb_loop(regression_methods, n_resamples = n_resamples, resample_dataset='train', predict_dataset='test', lambda_list = lambda_list)
    # score_list = ['mse']
    # plot_heatmap(lamb_scores, score_list = score_list, reg_method = 'ridge_own', title=f'Ridge regression with n_boots = {n_resamples}')

    # # =============== Ridge best ===============
    # # Best ridge: lambda = 0.001, poly_deg = 7
    # lambda_best = 0.001

    # ridge_best_scores = ra.bootstrap_loop(regression_methods = ['ridge_own'],
    #         n_resamples = n_resamples, resample_dataset='train', predict_dataset='test', lamb = lambda_best)

    # plot_bias_variance_tradeoff(ridge_best_scores, ['mse', 'bias', 'variance'], title=f'Ridge regression with $\lambda$ = {lambda_best}')


    # =============== Kfold Rdige lambda ===============
    # kfold_lamb_scores = ra.kfold_lamb_loop( regression_methods=['ridge_own'], n_splits=n_splits, dataset='data', lambda_list=lambda_list)
    # plot_heatmap(kfold_lamb_scores, ['mse'], reg_method='ridge_own', title=f'Ridge regression with KFold cross validation, k_folds = {n_splits}')

    # ===============  ===============
    # Lowest mse for kfold: lambda = 1e-5, deg = 6
    # kfold_labmda_best = 1e-5

    # =============== Lasso lambdas heaetmap ===============
    # warnings.filterwarnings('ignore')
    # lasso_lamb_scores = ra.boots_lamb_loop( ['lasso_skl'], n_resamples, 'train', 'test', lambda_list)
    # plot_heatmap( lasso_lamb_scores, ['mse'], 'lasso_skl', title=f'Lasso regression with Bootstrap, n_boots = {n_resamples}')
    

    # =============== Lasso kfold lambdas ===============
    # warnings.filterwarnings('ignore')
    # lasso_kfold_lamb_scores = ra.kfold_lamb_loop( regression_methods=['lasso_skl'], n_splits=n_splits, dataset='data', lambda_list=lambda_list)
    # plot_heatmap(lasso_kfold_lamb_scores, ['mse'], reg_method='lasso_skl', title=f'Lasso regression with KFold cross validation, k_folds = {n_splits}')

    
    # =============== Summary of best scores, ridge, lasso, mse (kfold, boots) ===============
    
    
    
    




