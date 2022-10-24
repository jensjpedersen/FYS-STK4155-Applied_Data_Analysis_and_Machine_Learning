from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import poly_data
import gradient_descent
import plot




if __name__ == '__main__':
    reload(poly_data)
    reload(gradient_descent)
    reload(plot)

    p = poly_data.PolyData()
    
    gd = gradient_descent.GradientDescent(p)
    beta_gd = gd.plane_gd(0.001, return_all=True)

    # Create dict with betas

    beta_dict = {'beta_gd': beta_gd}
    pl = plot.Plot(p)
    pl.plot_mse_vs_beta(beta_dict)






     

