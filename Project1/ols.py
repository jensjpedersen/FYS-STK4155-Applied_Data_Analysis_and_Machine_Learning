import numpy as np
import matplotlib.pyplot as plt
import sys
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression
import importlib
import franke_data
importlib.reload(franke_data)

def plot_franke_1d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    plt.show()


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
        linreg = LinearRegression(fit_intercept = True)
        linreg.fit(self.X_train, self.y_train)
        self.linreg = linreg
        beta = linreg.coef_
        beta[0] = linreg.intercept_
        self.beta = beta
        return self.beta

    def predict(self, X: np.ndarray):
        if self.method == 'skl': 
            return self.linreg.predict(X)

        return X @ self.beta




if __name__ == '__main__': 
    n = 4  # Poly deg
    N = 100 # dataset size
    f = franke_data.FrankeData(n, N, data_dim = 1)

    X_train, X_test, y_train, y_test = f.get_train_test_data(test_size = 0.2)
    o = OLS(X_train, y_train)


    # Ols training
    o.ols()
    y_tilde1 = o.predict(X_test)
    
    o.skl_ols()
    y_tilde2 = o.predict(X_test)
    


    fig = plt.figure()

    sys.exit()

    x = X_test[:, 1]
    y = X_test[:, 2]
    z = y_test
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='data')


    z = y_tilde1
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='own')

    z = y_tilde2
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='skl')
    plt.legend()
    plt.show()
    

