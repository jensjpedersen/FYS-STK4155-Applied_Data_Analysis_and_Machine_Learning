
from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
# import sklearn.ensemble
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import random_forest
import sklearn.utils
import descision_tree
import analyse_csv



import read_csv
from importlib import reload

import pandas as pd


def print_feature_ranking(clf, X, y): 
    # if isinstance(X, pd.DataFrame): 
    #     X = np.array(X)

    # if isinstance(y, pd.DataFrame): 
    #     y = np.array(y)

    clf.fit(X, y)

    # property feature_importances_ The impurity-based feature importances.
    # The higher, the more important the feature. The importance of a feature is
    # computed as the (normalized) total reduction of the criterion brought by that
    # feature. It is also known as the Gini importance.
    # Warning: impurity-based feature importances can be misleading for high
    # cardinality features (many unique values). See
    # sklearn.inspection.permutation_importance as an alternative.

    # importances = clf.feature_importances_

    importances = np.zeros((clf.n_estimators, X.shape[1]))

    for i, tree_clf in enumerate(clf.estimators_): 
        importances[i, :] = tree_clf.feature_importances_
        

    plt.figure(figsize=(12,8))
    sns.barplot(importances)#, errorbar='sd')
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    plt.show()

    importances = np.mean(importances, axis = 0)
    indices = np.argsort(importances)[::-1]
    # indices = indices[:top_features]

    features = clf.feature_names_in_


    feature_ranking = pd.DataFrame({'index': indices,
                                    'features': features[indices], 
                                    'Gini': importances[indices]})

    # print(feature_ranking)

    print(feature_ranking)
    return feature_ranking



def random_forest_sklearn(X, y): 
    clf = RandomForestClassifier(n_estimators = 10)
    # sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    clf.fit(X, y)

    trees = clf.estimators_
    n_tress = len(trees)

    values = []
    for i, tree in enumerate(trees): 
        val = tree.feature_importances_[1]
        values.append(val)


    print(np.mean(values))

    # Bootstrap row resampling, with equal size 

    # Choose random freatures. how to choose?


    # Train tree on data subset 


    # Feed all data trhrough forest, use maximum vote 



def plot_tree(X, y):
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    plt.figure(figsize=(16,8))
    tree.plot_tree(clf, fontsize=15)
    plt.show()


if __name__ == '__main__':
    reload(read_csv)
    reload(random_forest)
    reload(descision_tree)
    reload(analyse_csv)


    input_path = '../../Data/big_test.csv'
    r = read_csv.ReadCSV(input_path)
    r._remove_series([2, 3, 5])
    # r._remove_series([0, 2, 3, 5])
    X, y = r.get_df()



    a = analyse_csv.AnalyseCSV(r)
    a.plot_cluster_ravel(corr=True)
    plot_tree(X, y)


    # Sklearn 
    clf = RandomForestClassifier(n_estimators=100000, max_depth=None, max_features='sqrt')
    df_skl = print_feature_ranking(clf, X, y)

    # Own 
    clf = random_forest.RandomForest(n_estimators=100000, max_features='sqrt') 
    df_own = print_feature_ranking(clf, X, y)


    # print(df_skl['Gini'] / df_own['Gini'])


    







