import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
import terrain_data
import warnings
import resampling
from importlib import reload
import time
import pickle
import plot_terrain

reload(terrain_data)
reload(resampling)
reload(plot_terrain)

def dict_depth(scores):
        count = 0
        inner_dict = scores
        while count < 10: 
            outer_dict = inner_dict
            try: 
                first_key = list(outer_dict.keys())[0]
            except AttributeError: 
                break

            inner_dict = outer_dict[first_key]
            count += 1
        return count

def get_min_deg(scores: dict, score: str): 
    poly_deg_keys = list(scores.keys())
    poly_deg = [ int(deg) for deg in poly_deg_keys ]
    reg_method = scores['1'][score].keys()

    assert(len(reg_method) == 1)

    M_min = np.inf  
    for i, deg in enumerate(poly_deg_keys):
        M = scores[deg][score][list(reg_method)[0]]
        if M < M_min:
            M_min = M
            deg_min = deg

    print(f'{score} min: {M_min}, deg_min = {deg_min}')
    return M_min

def get_min_lamb(scores: dict, score: str):
    lambdas_keys = list(scores.keys())
    lambdas = [ float(lamb.replace('lamb:','')) for lamb in lambdas_keys ]
    poly_deg_keys = list(scores[lambdas_keys[0]])
    poly_deg = [ int(deg) for deg in poly_deg_keys ]
    reg_method = scores[lambdas_keys[0]]['1'][score].keys()


    assert(len(reg_method) == 1)

    M_min = np.inf  
    for j, lamb in enumerate(lambdas_keys): 
        for i, deg in enumerate(poly_deg_keys):
            M = scores[lamb][deg][score][list(reg_method)[0]]
            if M < M_min:
                M_min = M
                lambda_min = lamb
                deg_min = deg

    print(f'{score} min: {M_min}, lambda_min = {lambda_min}, deg_min = {deg_min}')



def plot_scores(scores: dict, score: str, legend: str = None, title: str = None): 
    """ Plots scores from scores dict in same plot"""
    # Get dict info
    poly_deg = np.array([ int(i) for i in scores ])
    reg_method = list(scores['1'][score].keys())[0]
    
    best_scores = [ scores[str(p)][score][reg_method] for p in poly_deg ]
    # score_names = score
    # method_names = list(scores['1'][score_names[0]].keys())

    # plt.plot(poly_deg, best_scores, label= f'{legend} {reg_method}')
    plt.plot(poly_deg, best_scores, label= f'{legend}')
    plt.legend()


    if title != None:
        plt.title(title)
        plt.xlabel('Polynomial degree')
        plt.ylabel(score.upper())
    # plt.xlabel('Polynomial degree')
    # plt.show()




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
    max_poly_deg = 30
    N_data = 30
    t = terrain_data.TerrainData(max_poly_deg, N_data)
    # t.plot_data()


    ra = resampling.ResamplingAnalysis(t)
    n_resamples = 20
    lambda_list = np.logspace(-6, 0, 7)
    n_splits = 10
    
    # =============== Boostrap OLS ===============

    #regression_methods = ['ols_own']
    #score_list = ['mse', 'bias', 'variance', 'r2']
    #ols_boots_scores = ra.bootstrap_loop(regression_methods = regression_methods,
            #n_resamples = n_resamples, resample_dataset='train', lamb = 1e-5, predict_dataset='test')


    # toc = time.perf_counter()
    # print(f'took: {toc-tic}')

    # score_list = ['mse', 'bias', 'variance']
    # resampling.plot_bias_variance_tradeoff(ols_boots_scores, score_list)
    # score_list = ['r2']
    # resampling.plot_bias_variance_tradeoff(ols_boots_scores, score_list)

    # =============== Lambda ridge bootstrap ===============
    #regression_methods = ['ridge_own']
    #score_list = ['mse']
    #ridge_boots_scores = ra.boots_lamb_loop(regression_methods, n_resamples = n_resamples, resample_dataset='train', predict_dataset='test', lambda_list = lambda_list)

    # resampling.plot_heatmap(ridge_boots_scores, deg_min=10, score_list = score_list, reg_method = 'ridge_own', title=f'Ridge regression with n_boots = {n_resamples}')


    # # =============== Ridge kfold ===============
    #kfold_ridge_scores = ra.kfold_lamb_loop( regression_methods=['ridge_own'], n_splits=n_splits, dataset='data', lambda_list=lambda_list)

    # resampling.plot_heatmap(kfold_ridge_scores, ['mse'], deg_min = 10, reg_method='ridge_own', title=f'Ridge regression with KFold cross validation, k_folds = {n_splits}')

    # =============== Lasso bootstrap ===============
    #regression_methods = ['lasso_skl']
    #score_list = ['mse']
    #warnings.filterwarnings('ignore')
    #lasso_boots_scores = ra.boots_lamb_loop(regression_methods, n_resamples = n_resamples, resample_dataset='train', predict_dataset='test', lambda_list = lambda_list)


    # resampling.plot_heatmap(lasso_boots_scores, deg_min=10, score_list = score_list, reg_method = 'lasso_skl', title=f'Lasso regression with n_boots = {n_resamples}')
    
    
    # =============== Lsso kfold ===============
    #kfold_lasso_scores = ra.kfold_lamb_loop( regression_methods=['lasso_skl'], n_splits=n_splits, dataset='data', lambda_list=lambda_list)
    # resampling.plot_heatmap(kfold_ridge_scores, ['mse'], deg_min = 10, reg_method='lasso_skl', title=f'Lasso regression with KFold cross validation, k_folds = {n_splits}')


    # =============== OLS kfold ===============

    #kfold_ols_scores = ra.kfold_lamb_loop( regression_methods=['ols_own'], n_splits=n_splits, dataset='data', lambda_list=lambda_list)
    
    
    # # ===============  best scores obtained with bootstrap ===============
    # boots_ols_scores = ra.bootstrap_loop(['ols_own'], 20, 'train', 'test', 0)
    # boots_ridge_scores = ra.bootstrap_loop(['ridge_own'], 20, 'train', 'test', 1e-6)
    # warnings.filterwarnings('ignore')
    # boots_asso_scores = ra.bootstrap_loop(['lasso_skl'], 20, 'train', 'test', 1e-6)

    # # ===============  best scores obtained with kfold ===============
    # kfold_ols_scores = ra.kfold_loop(['ols_own'], n_splits, 'data', 0)
    # kfold_ridge_scores = ra.kfold_loop(['ridge_own'], n_splits, 'data', 0.1)
    # warnings.filterwarnings('ignore')
    # kfold_lasso_scores = ra.kfold_loop(['lasso_skl'], n_splits, 'data', 0.001)


    # =============== Plot best scores ===============
    
    # fig = plt.figure(figsize=(12, 8))
    # plot_scores(kfold_ols_scores, 'mse', legend='OLS', title='MSE scores for best models with K-fold resampling')
    # plot_scores(kfold_ridge_scores, 'mse', legend='Ridge with $\lambda = 0.1$')
    # plot_scores(kfold_lasso_scores, 'mse', legend='Lasso with $\lambda = 0.001$')
    # plt.show()

    # fig = plt.figure(figsize=(12, 8))
    # plot_scores(boots_ols_scores, 'mse', legend='OLS', title='MSE scores for best models with Bootstrap resampling')
    # plot_scores(boots_ridge_scores, 'mse', legend='Ridge with $\lambda = 10^{-6}$')
    # plot_scores(boots_asso_scores, 'mse', legend='Lasso with $\lambda = 10^{-6}$')
    # plt.show()


    # =============== Plot terrain best model ===============
    # N_data = 30
    # t = terrain_data.TerrainData(18, N_data)
    # p = plot_terrain.PlotTerrain(t)
    # p.plot( method='ols_own')


    # ===============  Estimate error in Bootrap ===============
    # deg_min_list = []
    # for i in range(50):
    #     mse_boots_scores = ra.bootstrap_loop(['ols_own'], 20, 'train', 'test', 0)
    #     deg_min = get_min_deg(mse_boots_scores, 'mse')
    #     deg_min_list.append(deg_min)
        

