import poly_data
import copy
# import sklearn as skl
from sklearn.model_selection import train_test_split
from importlib import reload
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
import poly_data


if __name__ == '__main__':


    reload(poly_data)
    reload(neural_network)
    reload(scores)
    reload(activation)
    reload(optimizer)
    reload(analysis)

    np.random.seed(1)

# Load polynomial data
    p = poly_data.PolyData(2000)

    X_train, Y_train = p.get_train()
    X_test, Y_test = p.get_test()


    x_train, x_test = X_train[:,1][:,np.newaxis], X_test[:,1][:,np.newaxis]
    y_train, y_test = Y_train[:,np.newaxis], Y_test[:,np.newaxis]

    # x = np.linspace(0, 10, 1000)
    # y = x**2
    # x_train, x_test, y_train, y_test = train_test_split(x, y)


        
    parameters = {
        "X_train": x_train,
        "y_train": y_train,
        "X_test": x_test,
        "y_test": y_test,
        "eta": 0.001, 
        "depth": 2 ,
        "width": 5,
        "n_output_nodes": 1,
        "cost_score": 'mse',
        "activation_hidden": 'sigmoid',
        "activation_output": 'none',
        "gamma": 0.9,
        "lambd": 0.001,
        "tuning_method": 'none',
        "n_minibatches": 5,
        "epochs": 500 
        }


    # a = analysis.Analysis(**parameters)
    # a.eta = np.logspace(-2, -3, 2)
    # a.lambd = np.logspace(-1, -4, 4)
    # # a.plot_score('cost')

    # a.plot_heatmap('cost')

    a = analysis.Analysis(**parameters)
    a.labmd = 0.001
    a.eta = 0.01
    tn = a.setup_network(copy.deepcopy(a))
    y_pred_train = tn.get_output(x_train)
    y_pred_test = tn.get_output(x_test)
    plt.scatter(x_train.ravel(), y_pred_train.ravel(), label='trian lamb = 0.01')
    plt.scatter(x_test.ravel(), y_pred_test.ravel(), label='test lamb = 0.01')

    a = analysis.Analysis(**parameters)
    a.labmd = 0.001
    a.eta = 0.01
    tn = a.setup_network(copy.deepcopy(a))
    y_pred_train = tn.get_output(x_train)
    y_pred_test = tn.get_output(x_test)
    plt.scatter(x_train.ravel(), y_pred_train.ravel(), label='trian labm = 0.001')
    plt.scatter(x_test.ravel(), y_pred_test.ravel(), label ='test labm = 0.001')


    a = analysis.Analysis(**parameters)
    a.labmd = 0.001
    a.eta = 0.01
    tn = a.setup_network(copy.deepcopy(a))
    y_pred_train = tn.get_output(x_train)
    y_pred_test = tn.get_output(x_test)
    plt.scatter(x_train.ravel(), y_pred_train.ravel(), label='trian labm = 0.001')
    plt.scatter(x_test.ravel(), y_pred_test.ravel(), label ='test labm = 0.001')

    plt.scatter(x_train.ravel(), y_train.ravel(), label='model')
    plt.legend()
    plt.show()


    sys.exit()

    a = analysis.Analysis(**parameters)
    a.labmd = 0.001
    a.eta = 0.01
    tn = a.setup_network(copy.deepcopy(a))
    y_pred_train = tn.get_output(x_train)
    y_pred_test = tn.get_output(x_test)
    plt.scatter(x_train.ravel(), y_pred_train.ravel(), label='trian lamb = 0.01')
    plt.scatter(x_test.ravel(), y_pred_test.ravel(), label='test lamb = 0.01')

    a = analysis.Analysis(**parameters)
    a.labmd = 0.001
    a.eta = 0.001
    tn = a.setup_network(copy.deepcopy(a))
    y_pred_train = tn.get_output(x_train)
    y_pred_test = tn.get_output(x_test)
    plt.scatter(x_train.ravel(), y_pred_train.ravel(), label='trian labm = 0.001')
    plt.scatter(x_test.ravel(), y_pred_test.ravel(), label ='test labm = 0.001')

    plt.scatter(x_train.ravel(), y_train.ravel(), label='model')
    plt.legend()
    plt.show()
    





