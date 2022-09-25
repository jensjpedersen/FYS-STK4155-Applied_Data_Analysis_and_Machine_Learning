import numpy as np
import matplotlib.pyplot as plt
import sys
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression
import importlib
import franke_data
importlib.reload(franke_data)


@dataclass
class OLS: 
    X_train: np.ndarray = field(repr=False)
    y_train: np.ndarray = field(repr=False)

    beta: np.ndarray = field(init=False)

    method: str = field(init=False, default_factory=str)

    linreg: LinearRegression = field(init=False, default_factory=LinearRegression)


    def ols(self): 
        self.method = 'own'
        X_train = self.X_train
        y_train = self.y_train
        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        self.beta = beta
        return self.beta

    def skl_ols(self):
        self.method = 'skl'
        linreg = LinearRegression(fit_intercept = False)
        linreg.fit(self.X_train, self.y_train)
        self.linreg = linreg
        beta = linreg.coef_.copy()
        self.beta = beta

        return self.beta

    def predict(self, X: np.ndarray):
        if self.method == 'skl': 
            return self.linreg.predict(X)

        return X @ self.beta






if __name__ == '__main__': 
    n = 4  # Poly deg
    N = 10000 # dataset size
    test_size = 0.2
    f = franke_data.FrankeData(n, N, data_dim = 1, test_size=0.2)

    X_train, y_train = f.get_train()
    X_test = f.get_X_test()
    o = OLS(X_train, y_train)
    # Ols training
    o.ols()

    o.predict(X_test)

    # o.plot_model(X_test, data_dim = 1)
    # y_tilde1 = o.predict(X_test)
    
    # o.skl_ols()
    # y_tilde2 = o.predict(X_test)
    

    

