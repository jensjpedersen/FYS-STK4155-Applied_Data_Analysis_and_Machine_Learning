from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import neural_network
import copy
import optimizer 
import seaborn as sns


@dataclass
class Analysis: 
    X_train: np.ndarray = field(repr=False)
    y_train: np.ndarray = field(repr=False)

    X_test: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)

    # Network arcithecture
    depth: int
    width: int
    n_output_nodes: int

    cost_score: str
    activation_hidden: str
    activation_output: str

    eta: float
    gamma: float
    lambd: float

    epochs: int
    n_minibatches: int

    def plot_heatmap(self, score: str = 'accuracy'): 
        score_options = ['accuracy', 'mse']
        assert(score in score_options)

        self_copy = copy.deepcopy(self) # Don't want to do changes to origianl object
        values: list = []  # Heatmap is created from values in this list
                           # First tuple is plotted on x-axis 


        for key, val in vars(self).items(): 
            if key in ['X_train', 'y_train', 'X_test', 'y_test']:
                continue

            if not isinstance(val, list) and not isinstance(val, np.ndarray): 
                continue

            values.append((key, val))

        if len(values) != 2: 
            raise ValueError('Need two arrays/lists exactly')

        if len(values[1][1]) > len(values[0][1]): 
            values.reverse()

        print(values)


        # Loop through Values
        # set self variable

        train_scores = np.zeros( shape = (len(values[1][1]), len(values[0][1])) ) # Values with 
        test_scores = np.zeros_like(train_scores)

        x_label = values[0][0]
        y_label = values[1][0]


        for i, val_x in enumerate(values[0][1]): 
            for j, val_y in enumerate(values[1][1]): 
                setattr(self_copy, x_label, val_x)
                setattr(self_copy, y_label, val_y)
        
                # Setup neurla Network
                nn = neural_network.NeuralNetwork(
                    X_data = self_copy.X_train,
                    y_data = self_copy.y_train,
                    n_hidden_layers = self_copy.depth,
                    n_nodes_per_hidden_layer = self_copy.width,
                    n_output_nodes = self_copy.n_output_nodes,
                    cost_score = self_copy.cost_score,
                    activation_hidden = self_copy.activation_hidden,
                    activation_output = self_copy.activation_output
                    )


                op = optimizer.Optimizer(
                    eta = self_copy.eta,
                    gamma = self_copy.gamma,
                    lambd = self_copy.lambd
                    )

                tn = neural_network.TrainNetwork(nn, op, n_minibatches = self_copy.n_minibatches)

                tn.train(epochs=self_copy.epochs)

                # Add scores to matrix
                train_scores[j, i] = tn.get_accuracy(self.X_train, self.y_train)
                test_scores[j, i] = tn.get_accuracy(self.X_test, self.y_test)


        # Plot heatmaps
        plt.figure(figsize=(12,8))
        plt.title('Score on Training Data')
        sns.heatmap(train_scores, annot=True, fmt='.2f',
                # vmax = vmax, 
                cbar_kws={'label': 'Accuracy'}, 
                xticklabels = [str(x_val) for x_val in values[0][1]],
                yticklabels=[str(y_val) for y_val in values[1][1]]) 
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.figure(figsize=(12,8))
        plt.title('Score on Test Data')
        sns.heatmap(test_scores, annot=True, fmt='.2f',
                # vmax = vmax, 
                cbar_kws={'label': 'Accuracy'}, 
                xticklabels = [str(x_val) for x_val in values[0][1]],
                yticklabels=[str(y_val) for y_val in values[1][1]]) 

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.show()
        breakpoint() 












        
















    



