from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload

import logging
logging.basicConfig(format='%(message)s', filename='./test.log', encoding='utf-8', level=logging.DEBUG, force=True)

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=5,
            n_categories=1,
            epochs=5,
            # batch_size=100,
            eta=0.0001,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.X_data = X_data
        self.Y_data = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        # self.batch_size = batch_size
        # self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.counter = 0

        self.create_biases_and_weights()

        logging.info('INIT MHJ')

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):

        self.counter += 1
        print(self.counter)
        # feed-forward for training
        logging.info('=============== HiddenLayer.feed_forward ===============')
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = sigmoid(self.z_h)
        logging.info(f'z_l: ({self.z_h.shape}) = a_(l-1): ({self.X_data.shape}) @ W: ({self.hidden_weights.shape}) + b: ({self.hidden_bias.shape})')
        logging.info(f'W = {self.hidden_weights}')
        logging.info(f'b = {self.hidden_bias}')

        logging.info('=============== OutputLayer.feed_forward ===============')
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        # exp_term = np.exp(self.z_o)
        # self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        self.probabilities = self.z_o


        logging.info(f'z_l: ({self.z_o.shape}) = a_(l-1): ({self.a_h.shape}) @ W: ({self.output_weights.shape}) + b: ({self.output_bias.shape})')
        logging.info(f'W = {self.output_weights}')
        logging.info(f'b = {self.output_bias}')

    def feed_forward_out(self, X):
        logging.info('=============== feed_forward OUT - Never called OK ? ===============')
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)


        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        self.counter += 1
        print(self.counter)
        if len(self.Y_data.shape) == 1:
            self.Y_data = self.Y_data[:,np.newaxis]

        error_output = self.probabilities - self.Y_data
        # breakpoint() 
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            # for j in range(self.iterations):
            #     # pick datapoints with replacement
            #     chosen_datapoints = np.random.choice(
            #         data_indices, size=self.batch_size, replace=False
            #     )

            #     # minibatch training data
            #     self.X_data = self.X_data_full[chosen_datapoints]
            #     self.Y_data = self.Y_data_full[chosen_datapoints]

            self.feed_forward()
            self.backpropagation()
