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

    np.random.seed(2)
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


    parameters = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "eta": 0.001, 
        "depth": 1 ,
        "width": 10,
        "n_output_nodes": 1,
        "cost_score": 'cross_entropy',
        "activation_hidden": 'sigmoid',
        "activation_output": 'sigmoid',
        "gamma": 0.9,
        "lambd": 0,
        "tuning_method": 'none',
        "n_minibatches": 10,
        "epochs": 100
        }

    # Example: heatmap with eta and lambda
    # a = analysis.Analysis(**parameters)
    # a.eta = np.logspace(-5, -2, 4)
    # a.lambd = np.logspace(-5, -2, 4)
    # a.plot_heatmap()
    

    # Example: heatmap with width and depth
    # a = analysis.Analysis(**parameters)
    # a.width = [5, 10, 15, 20]
    # a.depth = [0, 1, 2, 3]
    # a.plot_heatmap()

    # Plot cost score for differnet tuning 
    # a = analysis.Analysis(**parameters)
    # a.tuning_method = ['none', 'rms_prop', 'adam']
    # a.plot_score('cost')

    # a = analysis.Analysis(**parameters)
    # a.activation_hidden = ['sigmoid', 'relu', 'leaky_relu']
    # # a.activation_hidden = 'relu'
    # # a.eta = np.logspace(-5, -1, 5)
    # # a.plot_heatmap()
    # # a.plot_score('cost')
    # a.epochs = 50
    # a.plot_score('accuracy')



#  ____            _         _ 
# |  _ \ __ _ _ __| |_    __| |
# | |_) / _` | '__| __|  / _` |
# |  __/ (_| | |  | |_  | (_| |
# |_|   \__,_|_|   \__|  \__,_|
                             
    # find optimal number of n_minibatches
    a = analysis.Analysis(**parameters)
    a.n_minibatches = [1, 5, 10, 15, 20]
    a.plot_score('accuracy')


    


