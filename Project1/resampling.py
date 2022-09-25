import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import sys
import sklearn as skl
import importlib
import franke_data
import ols 
import analysis 
importlib.reload(franke_data)
importlib.reload(ols)
importlib.reload(analysis)


@dataclass(frozen=True) 
class ResamplingAnalysis: 
    franke_object: franke_data.FrankeData

    X_train: np.ndarray = field(init=False, repr=False)
    X_test: np.ndarray = field(init=False, repr=False)
    y_train: np.ndarray = field(init=False, repr=False)
    y_test: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        train_test_split = self.franke_object.get_train_test_data()
        object.__setattr__(self, 'X_train', train_test_split[0])
        object.__setattr__(self, 'X_test', train_test_split[1])
        object.__setattr__(self, 'y_train', train_test_split[2])
        object.__setattr__(self, 'y_test', train_test_split[3])


    def calculate_scores_loop(self, regression_methods: list, n_resamples:int):
        max_poly_deg = self.franke_object.n
        resampling_scores = dict()
        # resampling_scores = {'mse': {}, 'bias': {}, 'variance': {}}
        
        for deg in range(1, max_poly_deg + 1): 
            resampling_scores[str(deg)] = {'mse': {}, 'bias': {}, 'variance': {}}

            l = int(((deg+1)*(deg+2)/2))		# Number of elements in beta
            _X_train = self.X_train[:, :l+1] 
            _X_test = self.X_test[:, :l+1]
            
            # bias variance tradeoff make sence on train data
            r = Resampling(_X_train, _X_test, self.y_train)

            for method in regression_methods: 
                y_boots_pred = r.bootstrap(n_resamples, method)
                rs = ResamplingScores(self.y_test, y_boots_pred)
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
        assert(np.size(self.y_test) == np.shape(self.y_pred)[0])
        if len(np.shape(self.y_pred)) != 2:
            raise ValueError('y_pred should be resampled in n samples')

    def bias(self): 
        y_test = self.y_test.reshape(np.shape(self.y_pred)[0], 1) # reshape array for subtraction to work 
        bias = np.mean( (y_test - np.mean(self.y_pred, axis=1, keepdims=True))**2 )
        return bias

    def variance(self): 
        y_test = self.y_test.reshape(np.shape(self.y_pred)[0], 1) # reshape array for subtraction to work 
        variance = np.mean( np.var(self.y_pred, axis=1, keepdims=True) )
        return variance

    def mse(self): 
        y_test = self.y_test.reshape(np.shape(self.y_pred)[0], 1) # reshape array for subtraction to work 
        mse = np.mean( np.mean((y_test - self.y_pred)**2, axis=1, keepdims=True) )
        return mse


@dataclass(frozen=True)
class Resampling: 
    franke_object: franke_data.FrankeData
    X_train: np.ndarray = field(init=False, repr=False)
    X_test: np.ndarray = field(init=False, repr=False)
    y_train: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        train_test_split = self.franke_object.get_train_test_data()
        object.__setattr__(self, 'X_train', train_test_split[0])
        object.__setattr__(self, 'X_test', train_test_split[1])
        object.__setattr__(self, 'y_train', train_test_split[2])
        # object.__setattr__(self, 'y_test', train_test_split[3])
        assert(np.size(self.y_train) == np.shape(self.X_train)[0])
        assert(np.shape(self.X_test)[0] < np.shape(self.X_train)[0])

    def kfold(self):
        pass

    def bootstrap(self, n_resamples: int, method: str):
        """
        Parameters:
            n_resamples: Number of resmaples
            method (str): Regression method that is used on test data.
                          Values: 'ols_skl', 'ols_own'

        Returns:
            y-pred: Matrix with predicted values of size:(n_test_data, n_resamples)
        """
        # Define trianing and test data
        X_test = self.franke_object.get_X_test()
        X_train, y_train = self.franke_object.get_train()

        y_pred = np.zeros((np.shape(X_test)[0], n_resamples))

        for i in range(n_resamples):
            X_, y_ = skl.utils.resample(X_train, y_train)
            y_pred[:,i] = self.regression_method(method, X_, y_, X_test)
        return y_pred

    def regression_method(self, method, X_train, y_train):
        X_ = X_train
        y_ = y_train

        if method == 'ols_skl': 
            o = ols.OLS(X_, y_)
            o.skl_ols()
            y_pred = o.predict(self.X_test)

        elif method == 'ols_own': 
            o = ols.OLS(X_, y_)
            o.ols()
            y_pred = o.predict(self.X_test)
        else:
            raise ValueError('Valide methods are: ols_skl, ols_own')

        return y_pred



def plot_bias_variance_tradeoff(scores): 
    # Get dict info
    poly_deg = np.array([ int(i) for i in scores ])
    score_names = list(scores['1'].keys())
    method_names = list(scores['1'][score_names[0]].keys())

    # Plot mse, bias and variance.  
    for method in method_names: 
        for score in score_names: 
            vals = np.array([ scores[str(deg)][score][method] for deg in poly_deg ])
            plt.plot(poly_deg, vals, label = f'{score} from {method}')

    plt.legend()
    plt.xlabel('Polynomial degree')
    plt.show()
            



if __name__ == '__main__': 
    np.random.seed(11)
    r = Resampling

    n = 12   # Poly deg
    N = 200 # dataset size
    n_resamples = 100
    noise = 0.05
    f = franke_data.FrankeData(n, N, data_dim = 1, add_noise = noise)

    X_train, X_test, y_train, y_test = f.get_train_test_data()


    max_poly_deg = n
    methods = ['ols_own']
    methods = ['ols_skl']
    methods = ['ols_own', 'ols_skl']

    ra = ResamplingAnalysis(f)
    scores = ra.calculate_scores_loop(n_resamples=n_resamples, regression_methods = methods)

    re = Resampling(f)
    # plot_bias_variance_tradeoff(scores)

    # resampling_methods = [boots, kfold_own, kfold_skl]



    # k-fold cross validtaion
    # shuffle the dataset
    # Split dataest into k groups
    # Choose one group as test data
    # Remaining gropus as training



