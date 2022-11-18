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
import tensorflow as tf
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

    np.random.seed(5)
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
        "n_minibatches": 20,
        "epochs": 400
        }


#  ____            _         _ 
# |  _ \ __ _ _ __| |_    __| |
# | |_) / _` | '__| __|  / _` |
# |  __/ (_| | |  | |_  | (_| |
# |_|   \__,_|_|   \__|  \__,_|
    # a = analysis.Analysis(**parameters)
    # # find optimal number of n_minibatches
    # a.epochs = 400
    # a.n_minibatches = [1, 5, 10, 15, 20]
    # a.plot_score('accuracy')

    # # Find optimal values fro lmabda and eta
    # a = analysis.Analysis(**parameters) # XXX: in progress
    # a.epochs = 200  # XXX: Reduced number of ephcos to speedup
    # a.lambd = np.logspace(-5, -1, 5)
    # a.lambd[0] = 0
    # a.eta = np.logspace(-5, -1, 5)
    # a.plot_heatmap('accuracy')


    # Use new best parameters for lambd and eta
    parameters = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "eta": 0.1,  # XXX: changed
        "depth": 1 ,
        "width": 10,
        "n_output_nodes": 1,
        "cost_score": 'cross_entropy',
        "activation_hidden": 'sigmoid',
        "activation_output": 'sigmoid',
        "gamma": 0.9,
        "lambd": 0.1,  # XXX: changed
        "tuning_method": 'none',
        "n_minibatches": 20,
        "epochs": 200 # XXX: changed
        }

    # # Example: heatmap with width and depth
    # a = analysis.Analysis(**parameters)
    # a.width = [5, 10, 15, 20]
    # a.depth = [1, 2, 3]
    # a.plot_heatmap('accuracy')


    # # Example: heatmap with width and depth
    # a = analysis.Analysis(**parameters)
    # a.lambd = 0
    # a.width = [5, 10, 15, 20]
    # a.depth = [1, 2, 3]
    # a.plot_heatmap('accuracy')


    # Use new parameters for network dept and width
    parameters = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "eta": 0.001,   # XXX: defualt leraning rate for adagrad and adam
        "depth": 2 ,  # XXX: changed 
        "width": 5,  # XXX: changed
        "n_output_nodes": 1,
        "cost_score": 'cross_entropy',
        "activation_hidden": 'sigmoid',
        "activation_output": 'sigmoid',
        "gamma": 0.9,
        "lambd": 0.1, 
        "tuning_method": 'none',
        "n_minibatches": 20,
        "epochs": 200 
        }


    # a = analysis.Analysis(**parameters)
    # a.lambd = 0
    # a.eta = 0.001 # Low eta to compare convergence rate
    # a.lambd = [0, 1e-3, 1e-2, 0.1]
    # a.plot_score('accuracy')




    # # # Test different tuning 
    # a = analysis.Analysis(**parameters)
    # # a.tuning_method = 'none'
    # a.gamma = 0.9
    # a.eta =0.001
    # a.lambd = 0
    # a.tuning_method = ['none', 'adam', 'rms_prop', 'adagrad']
    # a.plot_score('accuracy')

    # # # Test different tuning 
    # a = analysis.Analysis(**parameters)
    # # a.tuning_method = 'none'
    # a.gamma = 0
    # a.lambd = 0
    # a.eta = 0.001
    # a.tuning_method = ['none', 'adagrad', 'adam', 'rms_prop']
    # a.plot_score('accuracy')



    # Momentum vs tuning with best parms 
    # a = analysis.Analysis(**parameters)
    # a.tuning_method = 'none'
    # a.gamma = [0, 0.5, 0.9]
    # a.tuning_method = ['none', 'adam', 'rms_prop', 'adagrad']
    # a.plot_heatmap('accuracy')

    # # Test activation hidden layer 
    # a = analysis.Analysis(**parameters)
    # a.lambd = 0
    # a.gamma = 0
    # a.activation_hidden = ['sigmoid', 'relu', 'leaky_relu']
    # a.plot_score('accuracy')


    # Hatmap lambd vs activation hidden
    # a = analysis.Analysis(**parameters)
    # a.activation_hidden = ['sigmoid', 'relu', 'leaky_relu']
    # a.lambd = [0, 1e-3, 0.1]
    # a.plot_heatmap('accuracy')


    # Best vaues


    parameters = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "eta": 0.001,   # XXX: defualt leraning rate for adagrad and adam
        "depth": 2 ,  # XXX: changed 
        "width": 5,  # XXX: changed
        "n_output_nodes": 1,
        "cost_score": 'cross_entropy',
        "activation_hidden": 'sigmoid',
        "activation_output": 'sigmoid',
        "gamma": 0.9,
        "lambd": 0.01, 
        "tuning_method": 'adagrad',
        "n_minibatches": 20,
        "epochs": 300 
        }

    # a = analysis.Analysis(**parameters)
    # a.plot_score('accuracy') # acc: 0.9825, stable after 165 epochs  , eta = 0.001, gamma = 0.9, labmd = 0.01, adagrad 
    #                          # Worse performance with relu and leaky_relu

    # # Keras 
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Dense(5, activation=tf.nn.sigmoid))
    # model.add(tf.keras.layers.Dense(5, activation=tf.nn.sigmoid))
    # model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    # opt= tf.keras.optimizers.Adagrad(learning_rate=1)
    # # opt= tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # model.compile(optimizer=opt, loss='BinaryCrossentropy', metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=300, batch_size=len(y_train))
    # val_loss, val_acc = model.evaluate(X_test, y_test)
    # print(val_acc)



