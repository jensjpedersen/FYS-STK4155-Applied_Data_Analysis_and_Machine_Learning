from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import optimizer
from dataclasses import dataclass, field
import sys
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import poly_data
import numpy as np
import pprint
import time
import neural_network
import scores
from importlib import reload
import logging
import activation
import time
import optimizer
if __name__ == '__main__':


    reload(neural_network)
    reload(poly_data)
    reload(scores)
    reload(activation)
    reload(optimizer)

    np.random.seed(0)
    p = poly_data.PolyData(n_data=1000)

    depth = 1 
    width = 5 
    n_output_nodes = 1

    cost_score = 'mse'
    activation_hidden = 'sigmoid'
    activation_output = 'none'

    X_data, y_data = p.get_train()
    nn = neural_network.NeuralNetwork( X_data, y_data, depth, width, n_output_nodes, cost_score, activation_hidden, activation_output)

    eta = 0.001
    gamma = 0
    n_epochs = 1000
    n_minibatches = 1

    tuning_method = 'none'
    tuning_method = 'rms_prop'
    beta = 0.0
        
    op = optimizer.Optimizer(eta, gamma, tuning_method=tuning_method, beta=beta)
    tn = neural_network.TrainNetwork(nn, op, n_minibatches = n_minibatches)

    tic = time.perf_counter()
    tn.train(n_epochs)
    toc = time.perf_counter()
    print(f'took: {toc-tic}')


    x = p.get_X_trian()[:,1]

    y = nn.Layers[-1].get_output() # Not same
    y = tn.get_output(X_data) # as this. feed forward on updated weigts in last epochs

    plt.scatter(x, y, label='y')
    plt.scatter(x, y_data, label='t')
    plt.legend()
    plt.show()



    


