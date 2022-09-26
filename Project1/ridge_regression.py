import numpy as np
import matplotlib.pyplot as plt
import sys
from dataclasses import dataclass, field
import sklearn as skl
from sklearn.linear_model import Ridge
import importlib
import franke_data
importlib.reload(franke_data)


@dataclass
class RidgeRegression: 
    X_train: np.ndarray = field(repr=False)
    y_train: np.ndarray = field(repr=False)
    lamb: float

    beta: np.ndarray = field(init=False)

    method: str = field(init=False, default_factory=str)

    ridge: Ridge = field(init=False)

    def ridge_own(self): 
        self.method = 'own'
        X_train = self.X_train
        y_train = self.y_train
        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        n_features = np.shape(X_train)[1]
        I = np.eye(n_features)
        beta = np.linalg.pinv(X_train.T @ X_train + self.lamb*I) @ X_train.T @ y_train
        self.beta = beta
        return self.beta

    def ridge_skl(self):
        self.method = 'skl'
        ridge = Ridge(alpha = self.lamb, fit_intercept = False)
        ridge.fit(self.X_train, self.y_train)
        self.ridge = ridge
        beta = ridge.coef_.copy()
        self.beta = beta
        return self.beta

    def predict(self, X: np.ndarray):
        if self.method == 'skl': 
            return self.ridge.predict(X)

        return X @ self.beta






if __name__ == '__main__': 
    n = 4  # Poly deg
    N = 10000 # dataset size
    test_size = 0.2
    f = franke_data.FrankeData(n, N, data_dim = 1, test_size=0.2)

    o = RidgeRegression(f.get_X_train(), f.get_y_train(), lamb = 1)
    o.ridge_own()

    y_tilde = o.predict(f.get_X_test())

    # o.plot_model(X_test, data_dim = 1)
    # y_tilde1 = o.predict(X_test)
    
    # o.skl_ols()
    # y_tilde2 = o.predict(X_test)
    

    

