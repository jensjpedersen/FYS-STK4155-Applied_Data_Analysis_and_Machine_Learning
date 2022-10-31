from dataclasses import dataclass, field
import sys
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import poly_data
import numpy as np
import pprint
import time


@dataclass 
class NeuralNetwork:
    data_object: object 
    n_hidden_layers: int
    n_nodes_per_hidden_layer: int

    n_features: int = field(init=False)
    n_data: int = field(init=False)
    n_targets: int = field(init=False)

    X_data: np.ndarray = field(init=False, repr=False)
    y_data: np.ndarray = field(init=False, repr=False)

    Layers: list = field(default_factory=lambda: [])
    def __post_init__(self):
        self.X_data, self.y_data = self.data_object.get_train()

        self.n_features = self.X_data.shape[1]
        self.n_data = self.X_data.shape[0]
        self.n_targets = 10 # XXX: Automatize 


        self.__init_layers()


    def __init_layers(self): 
        # Init input layer
        self.Layers.append(self.__input_layer(self.n_features))

        # Init hidden layers
        for i in range(self.n_hidden_layers):
            self.Layers.append(self.__hidden_layer(self.n_nodes_per_hidden_layer))

        # Init ouput layer
        self.Layers.append(self.__output_layer(self.n_targets))


    def __hidden_layer(self, n_nodes: int): 
        """Returns hidden layer object"""
        W = np.random.randn(self.n_features, n_nodes)
        b = np.zeros(self.n_nodes_per_hidden_layer) + 0.01
        hl = HiddenLayer(W, b)
        return hl

    def __input_layer(self, n_nodes: int): 
        """Returns input layer object"""
        # Init hidden weigts and bias
        W = np.random.randn(self.n_features, n_nodes)
        b = np.zeros(self.n_nodes_per_hidden_layer) + 0.01
        il = InputLayer(W, b)
        return il

    def __output_layer(self, n_nodes: int):
        """Returns output layer object"""
        W = np.random.randn(self.n_features, n_nodes)
        b = np.zeros(self.n_nodes_per_hidden_layer) + 0.01
        ol = OuputLayer(W, b)
        return ol





@dataclass
class InputLayer:
    W: np.ndarray = field(repr=False)
    b: np.ndarray = field(repr=False)
    
@dataclass
class HiddenLayer:
    W: np.ndarray = field(repr=False)
    b: np.ndarray = field(repr=False)

    def update_weights(self): 
        pass

    def update_bias(self):
        pass


@dataclass
class OuputLayer:
    W: np.ndarray = field(repr=False)
    b: np.ndarray = field(repr=False)


