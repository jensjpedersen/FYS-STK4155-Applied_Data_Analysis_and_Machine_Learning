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

    af = activation.Activation(x, 'sigmoid')
    vals, der = af.get_values()
