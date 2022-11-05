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
import optimizer
import copy

logging.basicConfig(format='%(message)s', filename='./flow.log', encoding='utf-8', level=logging.DEBUG, force=True)
logging.getLogger().disabled = True
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
    def update_weights(initial_optimizer_ob: optimizer.Optimizer) -> None: 
        """ Calculates derivate of cost function with respect to weigts and bias for current layer """



@dataclass
class InputLayer(Layer):
    n_nodes: int # n_nodes = n_features
    X: np.ndarray = field(init=False, repr=False)  # Desing matrix with size (n_data, n_features)

    W = None
    b = None
    # W: np.ndarray = field(repr=False)
    # b: np.ndarray = field(repr=False)

    prev_Layer = None
    next_Layer = object()

    # update: bool = None # True when layers is ready for update, else False

    # def __post_init__(self): 

    def init_bias_and_weigths(self):
        pass

    def set_optimizer(self, op: optimizer.Optimizer): 
        return

    def get_output(self): 
        return self.X

    def feed_forward(self, X: np.ndarray):
        logging.info('================== InputLayer.feed_forward ===============')
        self.X = X
        # self.n_nodes = np.shape(self.X)[1]

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

    op: optimizer.Optimizer = field(init=False, default=None)

    def init_bias_and_weigths(self):
        # TODO: check if prev layer is Initialized
        self.b = np.zeros(self.n_nodes) + 0.01
        self.W = np.random.randn(self.prev_Layer.n_nodes, self.n_nodes)

    # def set_optimizer(self, op: optimizer.Optimizer): 
    #     if self.op != None: 
    #         raise AttributeError('Optimizer object is already initialized')

    #     self.op = op 

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

    def update_weights(self, initial_optimizer_ob: optimizer.Optimizer) -> None:
        logging.info('=============== HiddenLayer.update_weights ===============')
        if self.update != True:
            raise ValueError('Run method feed_forward, before atmepting to update weigts')
        pass

        if self.op == None: 
            assert(isinstance(initial_optimizer_ob, optimizer.Optimizer))
            self.op = copy.deepcopy(initial_optimizer_ob)

        delta = self.next_Layer.delta @ np.transpose(self.next_Layer.W) * self.derivative_activation
        gradient_weights = np.transpose(self.prev_Layer.get_output()) @ delta
        gradient_bias = np.sum(delta, axis = 0)

        # Use optimizer class to find new change
        W_change, b_change = self.op.update_change(gradient_weights, gradient_bias)

        W_new = self.W - W_change
        b_new = self.b - b_change

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
    n_nodes: int # n_nodes = n_targets
    score: str

    # t: np.ndarray = field(init=False, repr=False) # Targets

    W: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)

    prev_Layer: Layer = field(init=False, repr=False, default=lambda: object())
    next_Layer = None

    update: bool = None # True when layers is ready for update, else False

    a_l: np.ndarray = field(repr=False, default=None) # output from Activation function. Size (n nodes in layer l-) x (n_data)
                           # Initialize varablie with method: feed_forward
    # def __post_init__(self):
    op: optimizer.Optimizer = field(init=False, default=None)

    def init_bias_and_weigths(self): 
        self.b = np.zeros(self.n_nodes) + 0.01
        self.W = np.random.randn(self.prev_Layer.n_nodes, self.n_nodes)

    # def set_optimizer(self, op: optimizer.Optimizer): 
    #     if self.op != None: 
    #         raise AttributeError('Optimizer object is already initialized')

    #     self.op = op 

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

    def update_weights(self, initial_optimizer_ob: optimizer.Optimizer, targets: np.ndarray) -> None:

        logging.info('=============== OutputLayer.update_weights ===============')
        # TODO: need access to update sheme
        if self.update != True:
            raise ValueError('Run method feed_forward, before atmepting to update weigts')

        if self.op == None: 
            assert(isinstance(initial_optimizer_ob, optimizer.Optimizer))
            self.op = copy.deepcopy(initial_optimizer_ob)

        sc = scores.Scores(self.a_l, targets, self.score)  # Error are handled in Scores class
        grad_cost = sc.get_derivative()

        delta = self.derivative_activation * grad_cost 
        gradient_weights = np.transpose(self.a_l) @ delta
        gradient_bias = np.sum(delta, axis = 0)

        # Pass gradient
        # Calcaulate change in optimizer object
        W_change, b_change = self.op.update_change(gradient_weights, gradient_bias)

        # FIXME: if no optimizer, maybe? 
        # W_change = eta * gradient_weights
        # b_change = eta * gradient_bias

        W_new = self.W - W_change
        b_new = self.b - b_change

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
    X_data: np.ndarray = field(repr=False)
    y_data: np.ndarray = field(repr=False)

    n_hidden_layers: int
    n_nodes_per_hidden_layer: int
    # n_targets: int = field(init=False)
    n_output_nodes: int

    cost_score: str

    activation_hidden: str  # TODO
    activation_output: str  # TODO

    n_features: int = field(init=False)
    n_data: int = field(init=False)

    Layers: Layer = field(default_factory=lambda: [])
    op: optimizer.Optimizer = field(init=False, default=None)
    def __post_init__(self):
        X_data, y_data = self.X_data, self.y_data

        if len(X_data.shape) == 1:
            X_data = X_data[:,np.newaxis]

        if len(y_data.shape) == 1:
            y_data = y_data[:,np.newaxis]

        self.X_data, self.y_data = X_data, y_data
        self.n_features = self.X_data.shape[1]
        self.n_data = self.X_data.shape[0] # XXX: Problem ? 

        self.__init_layers()


    def __init_layers(self): 
        # Init input layer

        # Get input data
        n_features = self.X_data.shape[1]
        self.Layers.append(InputLayer(n_features)) 

        # Init hidden layers
        for i in range(self.n_hidden_layers):
            self.Layers.append(HiddenLayer( self.n_nodes_per_hidden_layer))

        # Init ouput layer
        self.Layers.append(OutputLayer(self.n_output_nodes, self.cost_score)) # XXX: remove cost score

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

    def set_optimizer(self, op: optimizer.Optimizer) -> None:
        self.op = op

    def get_optimizer(self) -> optimizer.Optimizer: 
        return self.op


@dataclass
class TrainNetwork: 
    """ Class used to train the network
    parameters:
        nn: NeuralNetwork object - Contains the arcithecture of the network
        op: Optimizer object - Contains shemes (gradient methods, etc.) for updateing weigts and biases 
    """
    nn: NeuralNetwork
    op: optimizer.Optimizer

    # Hyper parameters
    # gradient_method: str # sdg or gd 
    # eta: float # Learning rate
    # gamma: float = None

    t: np.ndarray = field(init=False, repr=False) # Targets
    a_L: np.ndarray = field(init=False, repr=False) # Values frm output Layer 

    def __post_init__(self):
        logging.info('=============== INIT NN ===============')
        self.t = self.nn.get_targets()
        assert(self.nn.op == None)
        self.nn.set_optimizer(self.op)
        assert(self.nn.op != None)


    def train(self, epochs: int): 
        for i in range(epochs):
            self.__feed_forward()
            self.__back_propagation()

            # print(i)
            print(self.get_score())

    def __feed_forward(self):
        X = self.nn.X_data # XXX Pass as arg

        layers = self.nn.get_layers()

        layers[0].feed_forward(X) # XXX: add x args

        for l in range(1, len(layers)-1): 
            layers[l].feed_forward(self.nn.activation_hidden)

        assert(isinstance(layers[l+1], OutputLayer))
        layers[l+1].feed_forward(self.nn.activation_output)
        output = layers[l+1].get_output()
        self.a_L = output

    def __back_propagation(self):
        targets = self.nn.get_targets()  # XXX: Add as arg 
        layers = self.nn.get_layers()
        op = self.op

        # Update weigts output layer
        assert(isinstance(layers[-1], OutputLayer))
        layers[-1].update_weights(op, targets)

        for l in range(len(layers)-2, 0, -1): 
            layers[l].update_weights(op)
            # XXX Optimizer needs update

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


    


