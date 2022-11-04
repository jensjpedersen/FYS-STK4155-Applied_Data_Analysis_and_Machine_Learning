import sys
import object_oriented
sys.path.append('../')
import poly_data
import numpy as np
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
from rich.traceback import install
install()

if __name__ == '__main__':

    reload(poly_data)
    reload(object_oriented)
    np.random.seed(0)

    p = poly_data.PolyData(n_data=1000)
    X, y = p.get_train()
    NN = object_oriented.NeuralNetwork( X, y, epochs=1000, eta=0.0001)
    NN.train()

    y = NN.z_o
    x = NN.X_data[:,1]
    t = NN.Y_data

    plt.scatter(x, y, label='y')
    plt.scatter(x, t, label='t')
    plt.legend()
    plt.show()




