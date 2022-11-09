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

if __name__ == '__main__':
    reload(neural_network)
    reload(scores)
    reload(activation)
    reload(optimizer)

    # Load the data
    cancer = load_breast_cancer()
    targets = cancer.target[:,np.newaxis]
    test_size = 0.2
    features = cancer.feature_names
    X_train, X_test, y_train, y_test = train_test_split(cancer.data,targets,random_state=0, test_size=test_size)
    print(X_train.shape)
    print(X_test.shape)

    # # Logistic Regression sklearn
    # logreg = LogisticRegression(solver='lbfgs')
    # logreg.fit(X_train, y_train)
    # print("Test set accuracy with Logistic Regression: {:.8f}".format(logreg.score(X_test,y_test)))


    # Scale data with mean and std
    scaler = skl.preprocessing.StandardScaler()
    scaler.fit(X_train)
    # X_train_scaled = scaler.transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    # np.random.seed(0)


    # eta = 0.00001
    eta = 0.001
    depth = 3 
    width = 10

    n_output_nodes = 1
    cost_score = 'mse'
    cost_score = 'cross_entropy'
    activation_hidden = 'sigmoid'
    activation_output = 'sigmoid'

    nn = neural_network.NeuralNetwork(X_train, y_train, depth, width, n_output_nodes, cost_score, activation_hidden, activation_output)

    gamma = 0.8
    lambd = 0
    # op = optimizer.Optimizer(eta, gamma)
    op = optimizer.Optimizer(eta, gamma, lambd=lambd)

    # op = optimizer.Optimizer(eta, 'gd')

    tn = neural_network.TrainNetwork(nn, op, n_minibatches = 10)

    tic = time.perf_counter()
    tn.train(200, True, X_test, y_test)
    # tn.train(200, True, )
    toc = time.perf_counter()
    y = tn.get_output(X_train)
    acc_train = tn.get_accuracy(X_train, y_train)
    acc_test = tn.get_accuracy(X_test, y_test)
    score_train = tn.get_score(X_train, y_train)
    score_test = tn.get_score(X_test, y_test)


    test_score = tn.get_all_test_scores()
    train_score = tn.get_all_train_scores()
    x = np.arange(1, len(test_score)+1)

    plt.scatter(x, test_score, label = 'test')
    plt.scatter(x, train_score, label = 'train')
    plt.legend()
    plt.show()

    test_acc = tn.get_all_test_accuracyies()
    train_acc = tn.get_all_train_accuracyies()

    plt.scatter(x, test_acc, label = 'test')
    plt.scatter(x, train_acc, label = 'train')
    plt.legend()
    plt.show()



    

    sys.exit()

    score = tn.scores_minibatch
    labels = np.arange(len(score))
    plt.plot(labels, score)
    plt.show()

     
    print(y)
    print(f'took: {toc-tic}')


    print(f'acc_train = {acc_train}')
    print(f'acc_test = {acc_test}')
    print(f'score_train = {score_train}')
    print(f'score_test = {score_test}')
    
    
    nn = neural_network.NeuralNetwork(X_train, y_train, depth, width, n_output_nodes, cost_score, activation_hidden, activation_output)
    op = optimizer.Optimizer(eta, gamma, lambd=lambd)
    tn = neural_network.TrainNetwork(nn, op, n_minibatches = 1)
    tn.train(100)



    y = tn.get_output(X_train)
    # acc_train = tn.get_accuracy(X_train, y_train)
    # acc_test = tn.get_accuracy(X_test, y_test)
    # score_train = tn.get_score(X_train, y_train)
    # score_test = tn.get_score(X_test, y_test)
    

    score = tn.scores_minibatch
    labels = np.arange(len(score))
    plt.plot(labels, score)
    plt.show()








    


