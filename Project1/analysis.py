import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

@dataclass
class Analysis: 
    x_data: np.ndarray
    y_data: np.ndarray 
    y_model: np.ndarray

    def r2(self):
        return 1 - np.sum((self.y_data - self.y_model) ** 2) / np.sum((self.y_data - np.mean(self.y_data)) ** 2)

    def mse(self):
        n = np.size(self.y_model)
        return np.sum((self.y_data-self.y_model)**2)/n

    def relative_errror(self):
        return abs((self.y_data-self.y_model)/self.y_data)

    def plot(self): 
        plt.figure()
        plt.plot(self.x_data, self.y_data, label = 'data')
        plt.plot(self.x_data, self.y_data, label = 'model')
        plt.show()


