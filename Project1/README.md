
# Info 
All files can be run as scripts. Examples of how to use the different
classes/functions can be found at the bottom of each file.

# Files
| Filename            | Description                                                     |
|---------------------|-----------------------------------------------------------------|
| analysis.py         | Class to extract mse and r2 scores, without re sampling         |
| franke_data.py      | Class used to generate dataset from Franke function             |
| lasso_regression.py | Class with functionally used for lasso regression               |
| main_franke.py      | Script used to produce results from Franke data in report       |
| main_terrain.py     | Script used to produce results from Terrain data in report      |
| ols.py              | Implementation of OLS regression method                         |
| plot_data.py        | Class for plotting of MSE, R2 and beta's                        |
| plot_model.py       | Class used to create surface plots of Franke data and model     |
| plot_terrain        | Class used to create surface plots of Terrain data and model    |
| resampling.py       | Implementation of K-fold and Bootstrap resampling               |
| ridge_regression.py | Implementation of our own and sklearn's Ridge Regression method |
| terrain_data.py     | Class used to generate dataset of Terrain in Oslo               |
| Oslo.tif            | File containing the terrain data                                |


# Reproducibility
In order to reproduce the data and figures in the report, uncomment the
different sections in main_franke.py and main_terrain.py 
 


