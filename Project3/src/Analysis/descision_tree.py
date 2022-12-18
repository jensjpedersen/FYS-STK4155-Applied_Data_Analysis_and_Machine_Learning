import numpy as np 
from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload


@dataclass
class Node:
    feature: int = None
    threshold: float = None  # Split value
    child_score: float = None  # 
    left: object = None   # Reference to left noe
    right: object = None  # Reference to right node
    depth: int = None
    information_gain: float = None # Loss in gini_index as result of split
    node_score: float = None
    class_count: np.ndarray = None # XXX: Only for leaf nodes. Count of class labels 


@dataclass
class DecisionTreeClassifier:
    max_depth: int = None # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split: int = 2  # The minimum number of samples required to split an internal node
    max_features: int | str = None # The number of features to consider when looking for the best split:
                                  # If None consider all features, str options = sqrt,
    root_node: Node = field(init=False, default=None) # Reference to root node

    n_samples: int = field(init=False, default=None)
    n_features: int = field(init=False, default=None)
    n_nodes: int = field(init=False, default=None)

    feature_importances_: np.ndarray = field(init=False, default=None)

    def fit(self, X, y):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(y, pd.DataFrame): 
            y = y.to_numpy()

        # Check that X and y have the same number of rows
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        # Create the root node of the decision tree
        if isinstance(self.root_node, type(None)):
            self.n_samples = X.shape[0]
            self.n_features = X.shape[1]
            self.n_nodes = 0
            self.root_node = self._build_tree(X, y, depth = 0)
            self._init_feature_importance()

    def predict(self, X):
        # Create a list to store the predictions
        predictions = []

        # For each row in the X matrix, make a prediction
        for i in range(X.shape[0]):
            prediction = self._predict_example(X[i, :], self.tree)
            predictions.append(prediction)

        return predictions


    def print_tree(self):

        def print_node(node, sign):
            print(sign)

            if isinstance(node.child_score, type(None)): 
                # print(f'{(node.depth+1)*"|        "}| depth: {node.depth}    child_score: {node.child_score}  {node.class_count}')
                print(f'{(node.depth+1)*"|        "}| depth: {node.depth}   node_score: {node.node_score}   {node.class_count}')
                return

            # print(f'{(node.depth+1)*"|        "}| depth: {node.depth}    child_score: {node.child_score}')
            print(f'{(node.depth+1)*"|        "}| depth: {node.depth}   node_score: {node.node_score}   ')

            print_node(node.right, f'{(node.depth+1)*"|--------"} Feature {node.feature} >= {node.threshold}    child_score: {node.child_score}    information_gain: {node.information_gain}')
            print_node(node.left, f'{(node.depth+1)*"|--------"} Feature {node.feature} < {node.threshold}    child_score: {node.child_score}    information_gain: {node.information_gain}')


        print_node(self.root_node, '|--------  root')



    def _init_feature_importance(self):
        feature_importance = np.zeros((self.n_nodes, self.n_features))
        counter = -1

        def recurse_tree(node, counter): 
            if isinstance(node.child_score, type(None)):
                return counter

            counter += 1

            feature_importance[counter, node.feature] = node.information_gain
            counter = recurse_tree(node.right, counter)
            counter = recurse_tree(node.left, counter)

            return counter


        counter = recurse_tree(self.root_node, counter)
        # self.feature_importances_ = np.mean(feature_importance, axis=0)
        self.feature_importances_ = np.max(feature_importance, axis=0)


    def _class_count(self, y): 
        """ return list with one hot vector and class count """
        class_count = []

        unique_classes = np.unique(y, axis=0)
        # if unique_classes.shape[0] > 1:
        for i in range(unique_classes.shape[0]):
            count = np.sum(unique_classes[i] == y, axis=0)[0]
            one_hot = unique_classes[i] 
            class_count.append([one_hot, count])
        # else:
        #     count = np.sum(unique_classes == y, axis=0)[0]
        #     one_hot = unique_classes
        #     class_count.append([one_hot, count])

        return class_count

    def _build_tree(self, X, y, depth):
        n_samples = X.shape[0]
        n_labels = np.unique(y, axis=0).shape[0]
        self.n_nodes += 1 


        if not isinstance(self.max_depth, type(None)) and depth >= self.max_depth: 
            return Node(depth=depth, class_count=self._class_count(y))

        elif n_samples < self.min_samples_split or n_labels == 1: 
            class_count = self._class_count(y)
            return Node(depth=depth, class_count=self._class_count(y))
        
        # FIXME: best_threshold is min value
        best_feature, best_threshold, best_score, information_gain, node_score = self._best_split(X, y)

        # Split the data into left and right subsets based on the best feature and value
        if isinstance(best_threshold, type(None)):
            breakpoint() 

        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]

        # if len(left_y) == 0 or len(right_y) == 0:
        #     # FIXME : problem with call to split:
        #     # Remove when done
        #     breakpoint() 

        left_tree = self._build_tree(left_X, left_y, depth + 1)
        right_tree = self._build_tree(right_X, right_y, depth + 1)

        node = Node(best_feature, best_threshold, best_score, left_tree, right_tree, depth, information_gain, node_score)
        # node = Node(1, 2, 3, )
        

        return node

    def _predict_example(self, x, tree):
        # If the current node is a leaf, return the class label
        if not isinstance(tree, dict):
            return tree

        # Select the left or right subtree based on the feature and value
        feature = list(tree.keys())[0]
        value = x[feature]
        subtree = tree[feature]['left'] if value < list(tree[feature].keys())[0] else tree[feature]['right']

        # Recursively traverse the subtree
        return self._predict_example(x, subtree)

    def _split(self, arr, threshold): 
        cond = arr < threshold
        left_indices = np.where(cond)
        right_indices = np.where(~cond)

        # if len(left_indices[0]) == 0:
        # #     # FIXME: remove when done
        #     breakpoint() 
        # if threshold == 0.74053:
        #     breakpoint()  

        return left_indices, right_indices

    def _feature_selection(self, X):
        # Bootstrap re-sampling of feature columns
        if isinstance(self.max_features, type(None)): 
            indices = np.arange(X.shape[1])

        elif isinstance(self.max_features, int): 
            indices = np.random.choice(np.arange(X.shape[1]), size=self.max_features, replace=False)

        elif isinstance(self.max_features, str) and self.max_features == 'sqrt': 
            sqrt = round(np.sqrt(X.shape[1])) 
            indices = np.random.choice(np.arange(X.shape[1]), size=sqrt, replace=False)

        else:
            raise ValueError(f' {self.max_features} is not a Valid option for self.max_features')

        return indices

    def _best_split(self, X, y):
        """
        Return: 
           best_feature: index of features that produces best split of current node 
           best_threshold: Feature value that gives best split 
           best_score: Weighed average gini index of child nodes
           information_gain: Reduction in gini_index after as result of split. 
        """


        # Initialize the best feature and value to split on
        best_feature, best_threshold = None, None
        best_score = np.inf # Best gini_index 

        # Gini index of current node   
        current_gini = self._gini_index(y)

        # Feature re-sampling
        feature_indices = self._feature_selection(X)
        # X = X[:, feature_indices]

        # For each feature and value pair
        # XXX: subbed range with indices
        for feature in feature_indices:
            sort_thresh = np.sort(np.unique(X[:, feature])) # XXX: remove sort when bug is fixed
            # for threshold in sort_thresh[1:-1]:
            for threshold in sort_thresh:


                # Split the data into left and right subsets based on the current feature and threshold
                left_indices, right_indices =  self._split(X[:, feature], threshold)
                left_y, right_y = y[left_indices], y[right_indices]


                left_gini = self._gini_index(left_y)
                rigth_gini = self._gini_index(right_y)

                # Calculate the weighted average
                weighted_avg_gini = left_y.shape[0] / y.shape[0] * left_gini + right_y.shape[0] / y.shape[0] * rigth_gini

                # If the weighted average entropy is lower than the current best score, update the best feature and value
                if weighted_avg_gini < best_score:
                    best_feature = feature
                    best_threshold = threshold
                    best_score = weighted_avg_gini

        weighted_information_gain = X.shape[0]/self.n_samples * (current_gini-best_score) 

        return best_feature, best_threshold, best_score, weighted_information_gain, current_gini # last param is information gain 


    def _entropy(self, y): 
        if y.shape[0] == 0: 
            return 0

        pdf = np.sum(y, axis=0)/y.shape[0]
        entropy = -np.sum(pdf*np.log(pdf))
        return entropy

    def _gini_index(self, y):
        if y.shape[0] == 0: 
            return 0

        pdf = np.sum(y, axis=0)/y.shape[0]
        # gini_index = 1 - np.sum(pdf**2)
        gini_index = np.sum(pdf*(1-pdf))
        return gini_index


    def _majority_class(self, y):
        # Return the majority class
        counts = {}
        for label in y:
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1
        return max(counts, key=counts.get)
