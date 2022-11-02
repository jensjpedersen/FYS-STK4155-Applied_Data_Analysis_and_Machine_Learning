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
from importlib import reload

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
        layer.feed_forward(stdout=True)





    

if __name__ == '__main__':
    reload(neural_network)

    reload(poly_data)
    p = poly_data.PolyData(n_data=100)

    depth = 2
    width = 20
    n_targets = 3
    nn = neural_network.NeuralNetwork(p, depth, width, n_targets)

    l1 = nn.get_layers(0)
    l1.feed_forward()
    l2 = nn.get_layers(1)
    l2.feed_forward()
    l3 = nn.get_layers(2)
    l3.feed_forward()
    l4 = nn.get_layers(3)
    l4.feed_forward()

    l4.update_weights()

    test_feed_forward(nn)
    # tn = neural_network.TrainNetwork(nn)
    # tn.feed_forward()

    


