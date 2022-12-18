import time
import numpy as np 
import sklearn.utils
import sklearn.tree
import read_csv
from importlib import reload
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import descision_tree
import pandas as pd


@dataclass
class RandomForest:
    n_estimators: int = 100
    max_depth: int = None
    min_samples_split: int = 2
    max_features: int = None

    estimators_: list = field(init=False, default_factory = lambda: [] )
    n_features: list = field(init=False, default_factory = lambda: [])
    feature_names_in_: list = field(init=False, default=None) # Initilized if X is pd.DataFrame

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        assert(X.shape[0] == y.shape[0])

        if isinstance(X, pd.DataFrame): 
            self.feature_names_in_ = np.array(X.columns)

        self.n_features = X.shape[1]

        trees: list = []
        for i in range(self.n_estimators):
            X_, y_ = sklearn.utils.resample(X, y)
            # clf = sklearn.tree.DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features)
            clf = descision_tree.DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features)
            clf.fit(X_, y_)
            trees.append(clf)
            if (i * 100) % self.n_estimators == 0:
                print(i/self.n_estimators*100)

        self.estimators_ = trees

    def get_best_features(self): 
        feature_importacne = np.zeros((self.n_estimators, self.n_features))

        for i, tree in enumerate(self.estimators_):
            feature_importacne[i, :] = tree.feature_importances_

        return np.mean(feature_importacne, axis=0)


if __name__ == '__main__':
    reload(read_csv)
    reload(descision_tree)

    # input_path = '../../Data/big_test.csv'
    # r = read_csv.ReadCSV(input_path)
    # r._remove_series([0, 2, 3, 5])

    # X, y = r.get_df()

    r = RandomForest(n_estimators=10, max_features='sqrt')
    r.fit(X,y)
    print(r.get_best_features())


