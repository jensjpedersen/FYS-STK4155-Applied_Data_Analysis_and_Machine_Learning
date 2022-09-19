import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression
import importlib
import franke_data
importlib.reload(franke_data)


@dataclass
class Regression: 
    X_train: np.ndarray = field(repr=False)
    X_test: np.ndarray = field(repr=False)
    y_train: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)

    beta: np.ndarray = field(init=False)
    y_tilde: np.ndarray = field(init=False)


    def ols(self): 
        X_train = self.X_train
        y_train = self.y_train
        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        y_tilde = X_train @ beta 
        self.beta = beta
        self.y_tilde = y_tilde
        return beta, y_tilde

    def skl_ols(self):
        linreg = LinearRegression(fit_intercept = True)
        linreg.fit(self.X_train, self.y_train)
        beta = linreg.coef_
        beta[0] = linreg.intercept_
        return beta
        # print('The intercept alpha: \n', linreg.intercept_)
        # print('Coefficient beta : \n', linreg.coef_)


if __name__ == '__main__':

    n = 5 # Poly deg
    N = 100 # dataset size
    f = franke_data.FrankeData(n, N)

    X_train, X_test, y_train, y_test = f.get_train_test_data(test_size = 0.2)
    o = Regression(X_train, X_test, y_train, y_test)

    # Ols training
    beta1, y1 = o.ols()
    linreg = o.skl_ols()


    

