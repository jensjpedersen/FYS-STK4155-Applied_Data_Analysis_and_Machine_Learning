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
class Resampling: 
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray

    def __post_init__(self):
        assert(np.size(self.y_train) == np.shape(self.X_train)[0])
        assert(np.shape(self.X_test)[0] < np.shape(self.X_train)[0])

    def bootstrap(self, n_resamples: int, method: str):
        y_pred = np.zeros((np.shape(X_test)[0], n_resamples))

        for i in range(n_resamples):
            X_, y_ = skl.utils.resample(self.X_train, self.y_train)

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

    # Ols training
    y_boots_pred = r.bootstrap(100, 'ols_own')

