from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import poly_data
import gradient_descent
import plot




if __name__ == '__main__':
    np.random.seed(0)
    reload(poly_data)
    reload(gradient_descent)
    reload(plot)

    eta = 0.001
    n_epochs = 50 
    n_epochs_sgd = 20
    p = poly_data.PolyData(n_data = 100)
    momentum = 0.1
    size_batch = 20
    
    gm = gradient_descent.GradientDescent(p)

    gm.gd(eta, n_epochs)
    gd_plain = gm.get_thetas()

    gm.gd(eta, n_epochs, momentum)
    # gm.gd(eta, n_epochs, 0.1) 
    gd_momentum = gm.get_thetas()

    gm.sgd( eta, n_epochs_sgd, size_batch)
    sgd = gm.get_thetas()

    gm.sgd( eta, n_epochs_sgd, size_batch, momentum)
    sgd_momentum = gm.get_thetas()

    # Plotting
    theta_dict = {'gd': gd_plain, 'gd_momentum': gd_momentum, 'sgd_theta': sgd, 'sgd_momentum': sgd_momentum}
    pl = plot.Plot(p)
    pl.plot_mse_vs_theta(theta_dict)







     

