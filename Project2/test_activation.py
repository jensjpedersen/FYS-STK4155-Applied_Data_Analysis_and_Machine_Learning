from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import activation
if __name__ == '__main__':
    reload(activation)

    x = np.linspace(-10, 5, 100)

    # af = activation.Activation(x, 'relu')
    af = activation.Activation(x, 'leaky_relu')
    vals, der = af.get_values()

    af = activation.Activation(x, 'relu')
    vals, der = af.get_values()
