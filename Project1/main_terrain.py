import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
import terrain_data
import warnings
import resampling
from importlib import reload
import time

reload(terrain_data)
reload(resampling)





    # # =============== Boots ===============
    # regression_methods = ['ols_own', 'ols_skl', 'ridge_own', 'ridge_skl', 'lasso_skl']
    # regression_methods = ['ols_own']
    # score_list = ['mse', 'bias', 'variance']
    # n_resamples = 100
    # boots_scores = ra.bootstrap_loop(regression_methods = regression_methods,
    #         n_resamples = n_resamples, resample_dataset='train', lamb = 0.0001, predict_dataset='train')
    # plot_bias_variance_tradeoff(boots_scores, score_list=score_list, 
    #         title = f'Bootstrap with n resamples = {n_resamples} and n datapoints = {n_data*n_data}')


if __name__ == '__main__':


    np.random.seed(1)
    tic = time.perf_counter()
    max_poly_deg = 30
    N_data = 30
    t = terrain_data.TerrainData(max_poly_deg, N_data)
    # t.plot_data()


    ra = resampling.ResamplingAnalysis(t)
    n_resamples = 20
    lambda_list = np.logspace(-6, 0, 7)
    n_splits = 10
    
    # =============== Boostrap OLS ===============

    # regression_methods = ['ols_own']
    # # regression_methods = ['ridge_own']
    # # # regression_methods = ['ridge_own', 'ridge_skl']
    # score_list = ['mse', 'bias', 'variance', 'r2']
    # boots_scores = ra.bootstrap_loop(regression_methods = regression_methods,
    #         n_resamples = n_resamples, resample_dataset='train', lamb = 1e-5, predict_dataset='test')

    # toc = time.perf_counter()
    # print(f'took: {toc-tic}')

    # score_list = ['mse', 'bias', 'variance']
    # resampling.plot_bias_variance_tradeoff(boots_scores, score_list)
    # score_list = ['r2']
    # resampling.plot_bias_variance_tradeoff(boots_scores, score_list)

    # # =============== Lambda ridge bootstrap ===============
    regression_methods = ['ridge_own']
    score_list = ['mse']
    lamb_boots_scores = ra.boots_lamb_loop(regression_methods, n_resamples = n_resamples, resample_dataset='train', predict_dataset='test', lambda_list = lambda_list)
    resampling.plot_heatmap(lamb_boots_scores, deg_min=10, score_list = score_list, reg_method = 'ridge_own', title=f'Ridge regression with n_boots = {n_resamples}')

    # # =============== Ridge kfold ===============
    # kfold_ridge_scores = ra.kfold_lamb_loop( regression_methods=['ridge_own'], n_splits=n_splits, dataset='data', lambda_list=lambda_list)
    # resampling.plot_heatmap(kfold_ridge_scores, ['mse'], deg_min = 10, reg_method='ridge_own', title=f'Ridge regression with KFold cross validation, k_folds = {n_splits}')

    # # =============== Lasso bootstrap ===============
    # regression_methods = ['lasso_skl']
    # score_list = ['mse']
    # warnings.filterwarnings('ignore')
    # lasso_boots_scores = ra.boots_lamb_loop(regression_methods, n_resamples = n_resamples, resample_dataset='train', predict_dataset='test', lambda_list = lambda_list)
    # resampling.plot_heatmap(lasso_boots_scores, deg_min=10, score_list = score_list, reg_method = 'lasso_skl', title=f'Lasso regression with n_boots = {n_resamples}')
    
    
    # =============== Lsso kfold ===============
    # kfold_ridge_scores = ra.kfold_lamb_loop( regression_methods=['lasso_skl'], n_splits=n_splits, dataset='data', lambda_list=lambda_list)
    # resampling.plot_heatmap(kfold_ridge_scores, ['mse'], deg_min = 10, reg_method='lasso_skl', title=f'Lasso regression with KFold cross validation, k_folds = {n_splits}')
    
