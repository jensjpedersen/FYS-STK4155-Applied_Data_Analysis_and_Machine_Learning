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
    tuning_method: str

    eta: float
    gamma: float
    lambd: float

    epochs: int
    n_minibatches: int

    def setup_network(self, self_copy, save_scores=False) -> neural_network.TrainNetwork: 
        """ Sets up optimizer and Network and trains network """

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
                lambd = self_copy.lambd,
                tuning_method = self_copy.tuning_method
                )

        tn = neural_network.TrainNetwork(nn, op, n_minibatches = self_copy.n_minibatches)

        if save_scores == True:
            tn.train(epochs=self_copy.epochs, save_scores = save_scores, X_test = self.X_test, y_test = self.y_test)
        else: 
            tn.train(epochs=self_copy.epochs)

        return tn

    def __get_scores(self, tn: neural_network.TrainNetwork, score):
        if score == 'accuracy':
            train_scores = tn.get_all_train_accuracyies()
            test_scores = tn.get_all_test_accuracyies()

        elif score == 'cost': 
            train_scores = tn.get_all_train_scores()
            test_scores = tn.get_all_test_scores()

        return train_scores, test_scores


    def plot_score(self, score: str): 
        """ Plot score as function of epochs """
        score_options = ['accuracy', 'cost']
        if not score in score_options: 
            raise ValueError(f'Spesify score, valid values are: {score_options}')

        self_copy = copy.deepcopy(self) 
        values = self.__find_list_variables()

        if len(values) > 1: 
            raise ValueError(f'Multiple variable lists is not supported. Got {len(values)}; {values}')


        fig, ax = plt.subplots()
        sns.set_style("darkgrid")
        plt.xlabel('Epochs')
        plt.ylabel(f'{score}')

        if score == 'cost': 
            plt.ylabel(f'{self.cost_score}')

        if len(values) == 0: 
            # if values is empty 
            tn = self.setup_network(self_copy, save_scores=True)
            train_scores, test_scores = self.__get_scores(tn, score)

            epochs = np.arange(1, len(train_scores)+1)
            sns.lineplot(x=epochs, y=test_scores, linewidth=1, label=f'test data')
            sns.lineplot(x=epochs, y=train_scores, linewidth=1, label=f'train data')
            self.toggle_legend(ax)
            plt.show()
            return

        variable_name = values[0][0]
        variable_values = values[0][1]
        for val in variable_values: 
            setattr(self_copy, variable_name, val)
            tn = self.setup_network(self_copy, save_scores = True)
            train_scores, test_scores = self.__get_scores(tn, score)

            epochs = np.arange(1, len(train_scores)+1)
            sns.lineplot(x=epochs, y=test_scores, linewidth=1, label=f'Test data, {variable_name}: {val}')
            sns.lineplot(x=epochs, y=train_scores, linewidth=1, label=f'Train data, {variable_name}: {val}')

        self.toggle_legend(ax)
        plt.show()


    def __find_list_variables(self) -> list: 
        """ 
        Returns:
            values: list of tuple[str, list]. First and second vlaue in tuple
                    correspond to variable name and values respectivly """
        values: list = []  # Heatmap is created from values in this list

        for key, val in vars(self).items(): 
            if key in ['X_train', 'y_train', 'X_test', 'y_test']:
                continue

            if not isinstance(val, list) and not isinstance(val, np.ndarray): 
                continue

            values.append((key, val))

        return values

    def plot_heatmap(self, score: str = 'accuracy'): 
        score_options = ['accuracy', 'cost'] # Cost not implemented
        assert(score in score_options)

        self_copy = copy.deepcopy(self) # Don't want to do changes to origianl object

        values = self.__find_list_variables()

        if len(values) != 2: 
            raise ValueError('Need two arrays/lists exactly')

        if len(values[1][1]) > len(values[0][1]): 
            values.reverse()

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
                tn = self.setup_network(self_copy)

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

    def toggle_legend(self, ax):
        leg = ax.legend(fancybox=True, shadow=True)

        lined = {}  # Will map legend lines to original lines.
        for legline, origline in zip(leg.get_lines(), ax.get_lines()):
            legline.set_picker(True)  # Enable picking on the legend line.
            lined[legline] = origline


        def on_pick(event):
            # On the pick event, find the original line corresponding to the legend
            # proxy line, and toggle its visibility.
            legline = event.artist
            origline = lined[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled.
            legline.set_alpha(1.0 if visible else 0.2)
            ax.get_figure().canvas.draw()

        ax.get_figure().canvas.mpl_connect('pick_event', on_pick)












        
















    



