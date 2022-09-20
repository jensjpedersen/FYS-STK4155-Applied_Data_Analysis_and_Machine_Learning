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

@dataclass
class ResamplingAnalysis: 
    y_test: np.ndarray
    y_pred: np.ndarray

    def __post_init__(self):
        assert(np.size(self.y_test) == np.shape(self.y_pred)[0])
        if len(np.shape(self.y_pred)) != 2:
            raise ValueError('y_pred should be resampled in n samples')

    def bias(self): 
        self.y_test = self.y_test.reshape(np.shape(self.y_pred)[0], 1) # reshape array for subtraction to work 
        bias = np.mean( (self.y_test - np.mean(self.y_pred, axis=1, keepdims=True))**2 )
        return bias

    def variance(self): 
        self.y_test = self.y_test.reshape(np.shape(self.y_pred)[0], 1) # reshape array for subtraction to work 
        variance = np.mean( np.var(self.y_pred, axis=1, keepdims=True) )
        return variance

    def mse(self): 
        self.y_test = self.y_test.reshape(np.shape(self.y_pred)[0], 1) # reshape array for subtraction to work 
        mse = np.mean( np.mean((self.y_test - self.y_pred)**2, axis=1, keepdims=True) )
        return mse


@dataclass
class Resampling: 
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray

    def __post_init__(self):
        assert(np.size(self.y_train) == np.shape(self.X_train)[0])
        assert(np.shape(self.X_test)[0] < np.shape(self.X_train)[0])

    def bootstrap(self, n_resamples: int, method: str):
        """
        Parameters:
            n_resamples: Number of resmaples
            method (str): Regression method that is used on test data.
                          Values: 'ols_skl', 'ols_own'

        Returns:
            Matrix with predicted values of size:(n_test_data, n_resamples)
        """
        y_pred = np.zeros((np.shape(X_test)[0], n_resamples))

        for i in range(n_resamples):
            X_, y_ = skl.utils.resample(self.X_train, self.y_train)

            # Analysis class goes in nice here
            if method == 'ols_skl': 
                o = ols.OLS(self.X_train, y_train)
                o.skl_ols()
                y_pred[:,i] = o.predict(self.X_test)

            elif method == 'ols_own': 
                o = ols.OLS(X_, y_)
                o.ols()
                y_pred[:,i] = o.predict(self.X_test)
            else:
                raise ValueError('Valide methods are: ols_skl, ols_own')

        return y_pred




if __name__ == '__main__': 
    r = Resampling

    n = 4  # Poly deg
    N = 10000 # dataset size
    f = franke_data.FrankeData(n, N, data_dim = 1)

    X_train, X_test, y_train, y_test = f.get_train_test_data(test_size = 0.2)
    r = Resampling(X_train, X_test, y_train)
    y_boots_pred = r.bootstrap(100, 'ols_own')

    ra = ResamplingAnalysis(y_test, y_boots_pred)
    mse = ra.mse()
    bias = ra.bias()
    variance = ra.variance()


