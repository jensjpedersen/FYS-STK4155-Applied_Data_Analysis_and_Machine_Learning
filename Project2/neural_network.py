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
import scores
from importlib import reload
import logging
import activation 

logging.basicConfig(format='%(message)s', filename='./flow.log', encoding='utf-8', level=logging.DEBUG, force=True)
# logging.getLogger('parso.python.diff').setLevel('INFO') 

# FORMAT = "%(message)s"
# # logging.basicConfig(
# #     filename=level=logging.DEBUG, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
# # )
# log = logging.getLogger("rich")
# log.info("Hello, World!")


# def SE(y, t):
#     """ Squared error cost function""" # XXX: removed 1/2
#     # assert(len(y.shape) == 1)
#     return jnp.sum((t - y)**2)

def costRidge(y, t, lamb): 
    # assert(len(y.shape) == 1)
    return jnp.sum((t - y)**2) + lamb * np.sum(y**2)

# def sigmoid(x):
#     return 1/(1 + jnp.exp(-x))

# def derivative_sigmoid(sigm): 
#     """ Input values gotten from sigmoid funcitno """
#     return sigm * (1 - sigm)


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
    def update_weights(eta: float, stdout=False) -> None: 
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

    # update: bool = None # True when layers is ready for update, else False

    def __post_init__(self): 
        self.n_nodes = np.shape(self.X)[1]

    def init_bias_and_weigths(self):
        pass

    def get_output(self): 
        return self.X

    def feed_forward(self, stdout:bool = False):
        logging.info('================== InputLayer.feed_forward ===============')

        logging.info(f'  X: ({self.X.shape})')

    def update_weights(self, eta, stdout=False) -> None:
        logging.info('=============== InputLayer.update_weights ===============')
        # Hidd # TODO: logging
        # self.update = False

    
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

    def feed_forward(self, activation_function: str) -> None:
        logging.info('=============== HiddenLayer.feed_forward ===============')
        output_prev_layer = self.prev_Layer.get_output()
        z_l = output_prev_layer @ self.W + self.b # Feed forward

        # Choose activation func
        sigma = activation.Activation(z_l, activation_function)
        self.a_l, self.derivative_activation = sigma.get_values()

        logging.info(f'z_l: ({z_l.shape}) = a_(l-1): ({output_prev_layer.shape}) @ W: ({self.W.shape}) + b: ({self.b.shape})')
        logging.info(f'W = {self.W}')
        logging.info(f'b = {self.b}')

    def update_weights(self, eta: float, stdout: bool = False) -> None:
        logging.info('=============== HiddenLayer.update_weights ===============')
        if self.update != True:
            raise ValueError('Run method feed_forward, before atmepting to update weigts')
        pass

        # error = self.t - self.a_l
        # gradient_weights = self.a_l.T @ error
        # gradietn_bias = error

        # delta = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)
        # hidden_weights_gradient = np.matmul(X.T, delta)
        # hidden_bias_gradient = np.sum(delta, axis=0)
    
        # error = 

        #                                                   XXX: dot or matrix proudct 

        # delta = self.derivative_activation * np.transpose(self.next_Layer.W) 
        # delta_next = self.next_Layer.delta
        delta = self.next_Layer.delta @ np.transpose(self.next_Layer.W) * self.derivative_activation
        # gradient_weights = delta * self.prev_Layer.get_output() 
        gradient_weights = np.transpose(self.prev_Layer.get_output()) @ delta
        gradient_bias = np.sum(delta, axis = 0)


        W_new = self.W - eta * gradient_weights
        b_new = self.b - eta * gradient_bias

        self.W = W_new
        self.b = b_new
        self.delta = delta

        self.update = False
        self.prev_Layer.update = True # Previous layer (l-1) is ready for update 

        logging.info(f'delta: ({delta.shape}) = next_Layer.delta: ({self.next_Layer.delta.shape}) @ transpose(next_Layer.W): ({np.transpose(self.next_Layer.W).shape}) * derivative_activation ({self.derivative_activation.shape})')
        logging.info(f'W_new: ({W_new.shape}) = W_old: ({self.W.shape}) - eta * gradient_weights: ({gradient_weights.shape})')
        logging.info(f'b_new: ({b_new.shape}) = b_old: ({self.b.shape}) - eta * gradient_weights: ({gradient_bias.shape})')
        # logging.info(f'W_new = {W_new}')
        # logging.info(f'b_new = {b_new}')



@dataclass
class OutputLayer(Layer):
    t: np.ndarray = field(repr=False) # Targets
    n_nodes: int # n_nodes = n_targets
    score: str

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

    def feed_forward(self, activation_output: str) -> None:
        logging.info('=============== OutputLayer.feed_forward ===============')
        if self.update == True: 
            raise ValueError('Network is ready for update. Run method update_weights before next iteration')

        output_prev_layer = self.prev_Layer.get_output()
        z_l = output_prev_layer @ self.W + self.b # Feed forward
        
        # Choose activation function
        ac = activation.Activation(z_l, activation_output)
        self.a_l, self.derivative_activation = ac.get_values()

        logging.info(f'z_l: ({z_l.shape}) = a_(l-1): ({output_prev_layer.shape}) @ W: ({self.W.shape}) + b: ({self.b.shape})')
        logging.info(f'W = {self.W}')
        logging.info(f'b = {self.b}')

        self.update = True

    def update_weights(self, eta: float, stdout: bool = False) -> None:
        logging.info('=============== OutputLayer.update_weights ===============')
        # TODO: need access to update sheme
        if self.update != True:
            raise ValueError('Run method feed_forward, before atmepting to update weigts')
        pass
        
        # TODO: choise between cost function
        # delta = self.t - self.a_l
        # print('Should eror')
        # if self.score == 'mse': 

        # grad_cost = grad(SE, 1)(self.a_l, self.t)

        sc = scores.Scores(self.a_l, self.t, self.score)  # Error are handled in Scores class
        # grad_cost = sc.get_score() # XXX OlD returns scalar
        grad_cost = sc.get_derivative()



        delta = self.derivative_activation * grad_cost 
        gradient_weights = np.transpose(self.a_l) @ delta
        gradient_bias = np.sum(delta, axis = 0)

        # TODO: Update Weigts 
        W_new = self.W - eta * gradient_weights
        b_new = self.b - eta * gradient_bias

        # print(W_new)
        # print(b_new)

        self.W = W_new
        self.b = b_new
        self.delta = delta

        self.update = False
        self.prev_Layer.update = True # Previous layer is ready for update

        logging.info(f'delta: ({delta.shape}) = derivative_activation: ({self.derivative_activation.shape}) * grad_cost: ({grad_cost.shape})')
        logging.info(f'W_new: ({W_new.shape}) = W_old: ({self.W.shape}) - eta * gradient_weights: ({gradient_weights.shape})')
        logging.info(f'b_new: ({b_new.shape}) = b_old: ({self.b.shape}) - eta * gradient_weights: ({gradient_bias.shape})')
        # logging.info(f'W_new = {W_new}')
        # logging.info(f'b_new = {b_new}')
        
        



@dataclass
class NeuralNetwork:
    """ Class with the basic arcithecture of the Network """
    data_object: object 
    n_hidden_layers: int
    n_nodes_per_hidden_layer: int
    # n_targets: int = field(init=False)
    n_output_nodes: int

    cost_score: str

    activation_hidden: str  # TODO
    activation_output: str  # TODO

    n_features: int = field(init=False)
    n_data: int = field(init=False)

    X_data: np.ndarray = field(init=False, repr=False)
    y_data: np.ndarray = field(init=False, repr=False)

    Layers: Layer = field(default_factory=lambda: [])
    def __post_init__(self):
        X_data, y_data = self.data_object.get_train()

        if len(X_data.shape) == 1:
            X_data = X_data[:,np.newaxis]

        if len(y_data.shape) == 1:
            y_data = y_data[:,np.newaxis]

        self.X_data, self.y_data = X_data, y_data
        self.n_features = self.X_data.shape[1]
        self.n_data = self.X_data.shape[0]

        self.__init_layers()


    def __init_layers(self): 
        # Init input layer

        # Get input data
        self.Layers.append(InputLayer(self.X_data))

        # Init hidden layers
        for i in range(self.n_hidden_layers):
            self.Layers.append(HiddenLayer( self.n_nodes_per_hidden_layer))

        # Init ouput layer
        targets = self.y_data
        self.Layers.append(OutputLayer(targets, self.n_output_nodes, self.cost_score))

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
    """ Class used to train the network
    parameters:
        nn: NeuralNetwork object - Contains the arcithecture of the network
        op: Optimizer object - Contains shemes (gradient methods, etc.) for updateing weigts and biases 
    """
    nn: NeuralNetwork = field(repr=False)

    # Hyper parameters
    eta: float # Learning rate

    t: np.ndarray = field(init=False, repr=False) # Targets
    a_L: np.ndarray = field(init=False, repr=False) # Values frm output Layer 

    def __post_init__(self):
        logging.info('=============== INIT NN ===============')
        self.t = self.nn.get_targets()

    def train(self, epochs: int): 
        for i in range(epochs):
            self.feed_forward()
            self.back_propagation()

            # print(i)
            print(self.get_score())

    def feed_forward(self):
        layers = self.nn.get_layers()

        for l, layer in enumerate(layers):
            if isinstance(layer, HiddenLayer):
                layer.feed_forward(self.nn.activation_hidden)
            elif isinstance(layer, OutputLayer):
                layer.feed_forward(self.nn.activation_output)

            # print(type(layer))
            output = layer.get_output()
            

        self.a_L = output

    def back_propagation(self):
        layers = self.nn.get_layers()
        for l in range(len(layers)-1, 0, -1): 
            layers[l].update_weights(self.eta)

    def get_output(self):
        try: 
            return self.a_L
        except AttributeError: 
            raise AttributeError('Use method "train" to create output variable (a_L)') 

    def get_score(self):
        sc = scores.Scores(self.a_L, self.t, self.nn.cost_score)
        return sc.get_score()

    def get_targets(self):
        return self.t


    


