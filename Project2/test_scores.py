from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import scores
import jax.numpy as jnp 
from jax import grad

def test_mse_derivative():
    x = np.linspace(0,1, 100)
    y = x**2 + x**3 + 5 + x**6 
    sc = scores.Scores(x, y)

    derivatives = sc.get_derivative()
    analytic = derivateve_MSE(x,y)
    
    assert(np.sum((analytic-derivatives)**2) < 1e-10)


def run_unit_tests(): 
    test_mse_derivative()

def derivateve_MSE(y, t): 
    return t-y  # M.H.J. definiton of mse XXX

if __name__ == '__main__':

    reload(scores)


    run_unit_tests()
    






