from dataclasses import dataclass, field
import sys
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import poly_data
import numpy as np
import pprint
import time
import neural_network
import scores
from importlib import reload
import logging
import activation
import time
import optimizer

def test_layers(nn: neural_network.NeuralNetwork):
    ' Check that layers are linked correct '

    layers = nn.get_layers()

    assert(layers[0].prev_Layer == None)
    assert(layers[-1].next_Layer == None)

    for i in range(len(layers)-1):
        assert(id(layers[i].next_Layer) == id(layers[i+1]))

    for i in range(1, len(layers)):
        assert(id(layers[i].prev_Layer) == id(layers[i-1]))

    print(' l:      prev_Layer    | layer     | next_Layer')
    for i, layer in enumerate(layers): 
        print(f' {i}: {id(layer.prev_Layer)} | {id(layer)} | {id(layer.next_Layer)}')

    print('Test layers OK')

def test_feed_forward(nn: neural_network): 
    " Prints size of matrices " 
    layers = nn.get_layers()
    for l, layer in enumerate(layers): 
        print(f'=============== FF Layer {l} ==================')
        layer.feed_forward(stdout=True)

def test_back_propagation(nn: neural_network):
    layers = nn.get_layers()
    eta = 0.001

    # for l, layer in enumerate(layers): 
    #     layer.feed_forward()

    for l in range(len(layers)-1, 0, -1): 
        print(f'=============== BP Layer {l} ==================')
        layers[l].update_weights(eta, stdout=True)





    

if __name__ == '__main__':


    reload(neural_network)
    reload(poly_data)
    reload(scores)
    reload(activation)
    reload(optimizer)

    np.random.seed(0)
    p = poly_data.PolyData(n_data=1000)

    depth = 1 
    width = 5 
    n_output_nodes = 1

    cost_score = 'mse'
    activation_hidden = 'sigmoid'
    activation_output = 'none'

    X_data, y_data = p.get_train()
    nn = neural_network.NeuralNetwork( X_data, y_data, depth, width, n_output_nodes, cost_score, activation_hidden, activation_output)

    eta = 0.0001
    gamma = 0
    n_epochs = 1000
    n_minibatches = 1

    op = optimizer.Optimizer(eta, gamma)
    # op = optimizer.Optimizer(eta, gamma, tuning_method='rms_prop', beta=0.9)
    # op = optimizer.Optimizer(eta, gamma, tuning_method='adam', beta1=0.9, beta2=0.99)
    tn = neural_network.TrainNetwork(nn, op, n_minibatches = n_minibatches)

    tic = time.perf_counter()
    tn.train(n_epochs)
    toc = time.perf_counter()
    print(f'took: {toc-tic}')


    x = p.get_X_trian()[:,1]

    y = nn.Layers[-1].get_output() # Not same
    y = tn.get_output(X_data) # as this. feed forward on updated weigts in last epochs

    plt.scatter(x, y, label='y')
    plt.scatter(x, y_data, label='t')
    plt.legend()
    plt.show()



    


