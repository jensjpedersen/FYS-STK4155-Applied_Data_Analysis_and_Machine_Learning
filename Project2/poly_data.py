from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt



@dataclass(frozen=True)
class PolyData: 
    x_range: tuple = (-1,1)
    n_data = 100
    coeff: tuple = (1, 0, 1)

    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self): 
        self.__create_datapoints()
        self.__create_design_matrix()
        self.__create_polynomial()

    def __create_datapoints(self): 
        x = np.linspace(self.x_range[0], self.x_range[1], self.n_data)
        object.__setattr__(self, 'x', x)

    def __create_polynomial(self): 
        y = self.X @ self.coeff
        object.__setattr__(self, 'y', y)
    
    def __create_design_matrix(self): 
        X = np.zeros((self.n_data, len(self.coeff)))  # Desing matrix
        for i in range(len(self.coeff)): 
            X[:,i] = self.x**i

        object.__setattr__(self, 'X', X)

    def plot(self):
        plt.figure()
        sns.set_style("darkgrid")
        sns.lineplot(x=self.x, y=self.y)
        plt.axis('equal')
        plt.show()


        

        
        # object.__setattr__(self, 'x', )
    

    




