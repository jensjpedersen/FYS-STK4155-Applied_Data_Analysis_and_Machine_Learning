import abc
from dataclasses import dataclass, field
import sys
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import poly_data
import numpy as np
import pprint
import time
import jax.numpy as jnp 
from jax import grad

def SE(y, t):
    """ Squared error cost function""" # XXX: removed 1/2
    # assert(len(y.shape) == 1)
    return jnp.sum((t - y)**2)

def costRidge(y, t, lamb): 
    # assert(len(y.shape) == 1)
    return jnp.sum((t - y)**2) + lamb * np.sum(y**2)

def sigmoid(x):
    return 1/(1 + jnp.exp(-x))

def derivative_sigmoid(sigm): 
    """ Input values gotten from sigmoid funcitno """
    return sigm * (1 - sigm)


# TODO: incremental id
@dataclass
class Layer(abc.ABC):

    @abc.abstractmethod
    def init_bias_and_weigths(): 
        " Initialize bias and weigts in layer "

    @abc.abstractmethod
    def get_output() -> np.ndarray:
        " Returns output from layer "

    @abc.abstractmethod
    def feed_forward(stdout: bool = False) -> None:
        " Calculates ouput from layer "

    @abc.abstractmethod
    def update_weights() -> None: 
        """ Calculates derivate of cost function with respect to weigts and bias for current layer """



@dataclass
class InputLayer(Layer):
    X: np.ndarray = field(repr=False)  # Desing matrix with size (n_data, n_features)
    n_nodes: int = field(init=False) # n_nodes = n_features

    W = None
    b = None
    # W: np.ndarray = field(repr=False)
    # b: np.ndarray = field(repr=False)

    prev_Layer = None
    next_Layer = object()

    update: bool = None # True when layers is ready for update, else False

    def __post_init__(self): 
        self.n_nodes = np.shape(self.X)[1]

    def init_bias_and_weigths(self):
        pass

    def get_output(self): 
        return self.X

    def feed_forward(self, stdout:bool = False):
        pass

    def update_weights(self) -> None:
        self.update = False

    
@dataclass
class HiddenLayer(Layer):
    n_nodes: int 

    W: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)

    prev_Layer: Layer = field(init=False, repr=False, default=lambda: object())
    next_Layer: Layer = field(init=False, repr=False, default=lambda: object())

    update: bool = None # True when layers is ready for update, else False

    a_l: np.ndarray = field(init=False, repr=False, default=None) # output from Activation function. Size (n nodes in layer l-) x (n_data)

    # Imparatn quantities for back back propagation
    derivative_activation: np.ndarray = field(init=False, repr=False) # Derivative of activation function with respect to z^l
    delta: np.ndarray = field(init=False, repr=False) # Derivate of Cost with respect to bias

    def init_bias_and_weigths(self):
        # TODO: check if prev layer is Initialized
        self.b = np.zeros(self.n_nodes) + 0.01
        self.W = np.random.randn(self.prev_Layer.n_nodes, self.n_nodes)

    def get_output(self):
        if np.size(self.a_l) == 1: 
            raise ValueError('Call method feed_forward to initlize variable')

        return self.a_l

    def feed_forward(self, stdout: bool = False) -> None:
        output_prev_layer = self.prev_Layer.get_output()
        z_l = output_prev_layer @ self.W + self.b # Feed forward

        # TODO: differnet activation function
        # If activation == 'sigmoid'
        a_l = sigmoid(z_l)  # Activation func
        self.a_l = a_l
        self.derivative_activation = derivative_sigmoid(a_l) # Calculate derivate of sigmoid funciton


        if stdout == True: 
            print(f'z_l: ({z_l.shape}) = a_(l-1): ({output_prev_layer.shape}) @ W: ({self.W.shape}) + b: ({self.b.shape})')

@dataclass
class OutputLayer(Layer):
    t: np.ndarray = field(repr=False) # Targets
    n_nodes: int # n_nodes = n_targets

    W: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)

    prev_Layer: Layer = field(init=False, repr=False, default=lambda: object())
    next_Layer = None

    update: bool = None # True when layers is ready for update, else False

    a_l: np.ndarray = field(repr=False, default=None) # output from Activation function. Size (n nodes in layer l-) x (n_data)
                           # Initialize varablie with method: feed_forward
    # def __post_init__(self):

    def __post_init__(self):
        if len(self.t.shape) == 1:
            self.t = self.t[:,np.newaxis]

    def init_bias_and_weigths(self): 
        self.b = np.zeros(self.n_nodes) + 0.01
        self.W = np.random.randn(self.prev_Layer.n_nodes, self.n_nodes)

    def get_output(self):
        if np.size(self.a_l) == 1: 
            raise ValueError('Call method feed_forward to initlize variable')

        return self.a_l

    def feed_forward(self, stdout: bool = False) -> None:
        if self.update == True: 
            raise ValueError('Network is ready for update. Run method update_weights before next iteration')

        output_prev_layer = self.prev_Layer.get_output()
        z_l = output_prev_layer @ self.W + self.b # Feed forward

        # TODO: option for activation function
        # Sigmoid as activation function for output layer
        a_l = sigmoid(z_l)  # Activation func
        self.derivative_activation = derivative_sigmoid(a_l)
        self.a_l = a_l

        if stdout == True: 
            print(f'z_l: ({z_l.shape}) = a_(l-1): ({output_prev_layer.shape}) @ W: ({self.W.shape}) + b: ({self.b.shape})')

@dataclass 
class NeuralNetwork:
    data_object: object 
    n_hidden_layers: int
    n_nodes_per_hidden_layer: int
    # n_targets: int = field(init=False)
    n_output_nodes: int

    n_features: int = field(init=False)
    n_data: int = field(init=False)

    X_data: np.ndarray = field(init=False, repr=False)
    y_data: np.ndarray = field(init=False, repr=False)

    Layers: Layer = field(default_factory=lambda: [])
    def __post_init__(self):
        self.X_data, self.y_data = self.data_object.get_train()

        self.n_features = self.X_data.shape[1]
        self.n_data = self.X_data.shape[0]

        self.__init_layers()


    def __init_layers(self): 
        # Init input layer

        # Get input data
        self.Layers.append(InputLayer(self.X_data))

        # Init hidden layers
        for i in range(self.n_hidden_layers):
            print(f'i = {i}')
            print(f'selv.n_nodes_per_hidden_layer = {self.n_nodes_per_hidden_layer}')
            
            self.Layers.append(HiddenLayer( self.n_nodes_per_hidden_layer))

        # self.Layers.append(OutputLayer( self.n_targets))
        targets = self.y_data
        self.Layers.append(OutputLayer(targets, self.n_output_nodes))

        # Connect layers
        for i, Layer in enumerate(self.Layers):
            if Layer.next_Layer != None:
                Layer.next_Layer = self.Layers[i+1]

            if Layer.prev_Layer != None:
                Layer.prev_Layer = self.Layers[i-1]

        # Initialize bias and weigts
        for i, layer in enumerate(self.Layers): 
            layer.init_bias_and_weigths()


    def get_layers(self, l: int = None):
        if l == None: 
            return self.Layers
        return self.Layers[l]

    def get_targets(self): 
        return self.y_data


@dataclass
class TrainNetwork: 
    nn: NeuralNetwork = field(repr=False)

    t: np.ndarray = field(init=False, repr=False) # Targets
    a_L: np.ndarray = field(init=False, repr=False) # Values frm output Layer 

    def __post_init__(self):
        self.t = self.nn.get_targets()


    def feed_forward(self):
        layers = self.nn.get_layers()

        for l, layer in enumerate(layers):
            layer.feed_forward()
            output = layer.get_output()

        self.a_L = output

    def back_propagation(self): 
        pass


    


