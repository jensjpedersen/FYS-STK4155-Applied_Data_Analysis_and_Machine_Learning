from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import time

from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import sklearn as skl 

import neural_network
import scores
import activation
import optimizer
import analysis

if __name__ == '__main__':
    reload(neural_network)
    reload(scores)
    reload(activation)
    reload(optimizer)
    reload(analysis)

    # Load the data
    cancer = load_breast_cancer()
    targets = cancer.target[:,np.newaxis]
    test_size = 0.2
    features = cancer.feature_names
    X_train, X_test, y_train, y_test = train_test_split(cancer.data,targets,random_state=0, test_size=test_size)

    # Scale data with mean and std
    scaler = skl.preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # np.random.seed(0)

    parameters = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "eta": 0.001, 
        "depth": 3 ,
        "width": 10,
        "n_output_nodes": 1,
        "cost_score": 'cross_entropy',
        "activation_hidden": 'sigmoid',
        "activation_output": 'sigmoid',
        "gamma": 0.8,
        "lambd": 0,
        "n_minibatches": 1,
        "epochs": 500
        }

    a = analysis.Analysis(**parameters)
    a.eta = [1, 0.1, 0.001]
    a.eta = np.logspace(-5, -2, 4)
    a.lambd = np.logspace(-5, -2, 4)

    a.plot_heatmap()
    



    


