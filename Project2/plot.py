from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload

def mse(y_test, y_pred):
    return np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))

@dataclass
class Plot:
    data_object: object


    def plot_mse_vs_beta(self, beatas: dict, dataset: str ='train'): 
        get_func_name = f'get_{dataset}'
        get_data = getattr(self.data_object, get_func_name)
        X, y = get_data()
        y = y.reshape(-1, 1)

        for method, betas in beatas.items():

            mse_scores = np.zeros(len(betas))

            for i in range(np.shape(betas)[0]):
                beta = betas[i,:].reshape(-1, 1)
                mse_scores[i] = mse(y, X@beta)

            for i in range(np.shape(betas)[1]):
                plt.figure()
                sns.set_style("darkgrid")
                plt.xlabel(f'$\\beta_{i}$')
                plt.ylabel('MSE')
                sns.scatterplot(x=betas[:,i], y=mse_scores)
            plt.show()









