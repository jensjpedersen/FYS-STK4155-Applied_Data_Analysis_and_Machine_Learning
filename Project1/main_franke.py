import numpy as np
import matplotlib.pyplot as plt
import franke_data
import resampling
import analysis
import plot_data
import plot_model
from importlib import reload 
reload(franke_data)
reload(resampling)
reload(analysis)
reload(plot_data)

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
        plt.ylabel('MSE')
    # plt.xlabel('Polynomial degree')
    # plt.show()

if __name__ == '__main__':

    # TODO, compare own kfold VS sklearn
    np.random.seed(1)

    max_poly_deg = 12
    n_data = 20
    test_size = 0.2
    noise = 0.2
    data_dim = 2

    f = franke_data.FrankeData(max_poly_deg, n_data, data_dim = data_dim, add_noise = noise, test_size = test_size, set_seed=True)
    ra = resampling.ResamplingAnalysis(f)

    # =============== Resampling parameters ===============
    n_resamples = 100
    lambda_list = np.logspace(-6, 1, 8)
    n_splits = 10


    kfold_scores = ra.kfold_loop(['ols_own'], n_splits, 'data', 0) 
    resampling.plot_bias_variance_tradeoff(kfold_scores, ['mse'])

    # # =============== Kfold ===============
    # regression_methods = ['ols_own']
    # kfold_scores = ra.kfold_loop(regression_methods = regression_methods, n_splits=n_splits, dataset='data', lamb = 0.001)
    # score_list = ['mse']
    # resampling.plot_bias_variance_tradeoff(kfold_scores, score_list=score_list, title = f'Kfold cross validation with n = {n_splits} splits')

    # # =============== Boots ===============
    # score_list = ['mse', 'bias', 'variance']
    # boots_scores = ra.bootstrap_loop(regression_methods = regression_methods,
    #         n_resamples = n_resamples, resample_dataset='train', lamb = 0.0001, predict_dataset='test')
    # resampling.plot_bias_variance_tradeoff(boots_scores, score_list=score_list, 
    #         title = f'Bootstrap with n resamples = {n_resamples} and n datapoints = {n_data*n_data}')

    # # =============== Boots lambdas heatmap ===============
    # regression_methods = ['ridge_own']
    # lamb_scores = ra.boots_lamb_loop(regression_methods, n_resamples = n_resamples, resample_dataset='train', predict_dataset='test', lambda_list = lambda_list)
    # score_list = ['mse']
    # resampling.plot_heatmap(lamb_scores, score_list = score_list, reg_method = 'ridge_own', title=f'Ridge regression with n_boots = {n_resamples}')

    # # =============== Ridge best ===============
    # # Best ridge: lambda = 0.001, poly_deg = 7
    # lambda_best = 0.001

    # ridge_best_scores = ra.bootstrap_loop(regression_methods = ['ridge_own'],
    #         n_resamples = n_resamples, resample_dataset='train', predict_dataset='test', lamb = lambda_best)

    # resampling.plot_bias_variance_tradeoff(ridge_best_scores, ['mse', 'bias', 'variance'], title=f'Ridge regression with $\lambda$ = {lambda_best}')

    # =============== Kfold Rdige lambda ===============
    # kfold_lamb_scores = ra.kfold_lamb_loop( regression_methods=['ridge_own'], n_splits=n_splits, dataset='data', lambda_list=lambda_list)
    # resampling.plot_heatmap(kfold_lamb_scores, ['mse'], reg_method='ridge_own', title=f'Ridge regression with KFold cross validation, k_folds = {n_splits}')

    # ===============  ===============
    # Lowest mse for kfold: lambda = 1e-5, deg = 6
    # kfold_labmda_best = 1e-5

    # =============== Lasso lambdas heaetmap ===============
    # warnings.filterwarnings('ignore')
    # lasso_lamb_scores = ra.boots_lamb_loop( ['lasso_skl'], n_resamples, 'train', 'test', lambda_list)
    # resampling.plot_heatmap( lasso_lamb_scores, ['mse'], 'lasso_skl', title=f'Lasso regression with Bootstrap, n_boots = {n_resamples}')
    

    # =============== Lasso kfold lambdas ===============
    # warnings.filterwarnings('ignore')
    # lasso_kfold_lamb_scores = ra.kfold_lamb_loop( regression_methods=['lasso_skl'], n_splits=n_splits, dataset='data', lambda_list=lambda_list)
    # resampling.resampling..plot_heatmap(lasso_kfold_lamb_scores, ['mse'], reg_method='lasso_skl', title=f'Lasso regression with KFold cross validation, k_folds = {n_splits}')

    
    # =============== Summary of best scores, ridge, lasso, mse (kfold, boots) ===============

    # =============== Plot best mse scores ===============
    
    # # Plot best scores obtained with bootstrap
    

    # ridge_boots_scores = ra.bootstrap_loop(['ridge_own'], 100, 'train', 'test', 0.0001)
    # lasso_boots_scores = ra.bootstrap_loop(['lasso_skl'], 100, 'train', 'test', 1e-5)


    # # # Best mse scores obtianed with kfold
    # mse_kfold_scores = ra.kfold_loop(['ols_own'], 100, 'data', 0)
    # ridge_kfold_scores = ra.kfold_loop(['ridge_own'], 100, 'data', 1e-5)
    # lasso_kfold_scores = ra.kfold_loop(['lasso_skl'], 100, 'data', 1e-5)

    # fig = plt.figure(figsize=(12, 8))
    # plot_scores(mse_boots_scores, 'mse', 'OLS', 'MSE scores for best models with Bootstrap resmapling')
    # plot_scores(ridge_boots_scores, 'mse', 'Ridge with $\lambda = 10^{-4}$')
    # plot_scores(lasso_boots_scores, 'mse', 'Lasso with $\lambda = 10^{-5}$')
    # plt.show()

    # fig = plt.figure(figsize=(12, 8))
    # plot_scores(mse_boots_scores, 'mse', 'OLS', 'MSE scores for best models with K-fold resmapling')
    # plot_scores(ridge_boots_scores, 'mse', 'Ridge with $\lambda = 10^{-5}$')
    # plot_scores(lasso_boots_scores, 'mse', 'Lasso with $\lambda = 10^{-5}$')
    # plt.show()

    # =============== Plot model ===============

    # # ===============  Estimate error in Bootrap ===============
    # deg_min_list = []
    # for i in range(50):
    #     mse_boots_scores = ra.bootstrap_loop(['ols_own'], 100, 'train', 'test', 0)
    #     deg_min = get_min_deg(mse_boots_scores, 'mse')
    #     deg_min_list.append(deg_min)


