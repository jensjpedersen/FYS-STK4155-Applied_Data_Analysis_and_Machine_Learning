import numpy as np
import matplotlib.pyplot as plt
import sys
from dataclasses import dataclass, field
import sklearn as skl
from sklearn.linear_model import Lasso
import importlib
import franke_data
importlib.reload(franke_data)


@dataclass
class LassoRegression: 
    X_train: np.ndarray = field(repr=False)
    y_train: np.ndarray = field(repr=False)
    lamb: float

    beta: np.ndarray = field(init=False)

    method: str = field(init=False)

    lasso: Lasso = field(init=False)

    def lasso_skl(self):
        self.method = 'skl'
        lasso = Lasso(alpha = self.lamb, fit_intercept = False)
        lasso.fit(self.X_train, self.y_train)
        self.lasso = lasso
        beta = lasso.coef_.copy()
        self.beta = beta
        return self.beta

    def predict(self, X: np.ndarray):
        return self.lasso.predict(X)







if __name__ == '__main__': 
    n = 6  # Poly deg
    N = 10000 # dataset size
    test_size = 0.2
    f = franke_data.FrankeData(n, N, data_dim = 1, test_size=0.2)

    X_train, y_train = f.get_train()
    X_test = f.get_X_test()

    # lasso = Lasso(alpha = 0.1, fit_intercept = False)
    # lasso.fit(X_train, y_train)
    # y_tidle = lasso.predict(X_test)
    # sys.exit()


    o = LassoRegression(X_train, y_train, lamb = 0.1)
    o.lasso_skl()

    y_tilde = o.predict(f.get_X_test())

    # o.plot_model(X_test, data_dim = 1)
    # y_tilde1 = o.predict(X_test)
    
    # o.skl_ols()
    # y_tilde2 = o.predict(X_test)
    

    

