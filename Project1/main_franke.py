import numpy as np
import franke_data
import resampling
import analysis
import plot_data
from importlib import reload 
reload(franke_data)
reload(resampling)
reload(analysis)
reload(plot_data)

if __name__ == '__main__':
    # np.random.seed(1)

    max_poly_deg = 18
    n_data = 20
    test_size = 0.2
    noise = 0.2
    data_dim = 2

    f = franke_data.FrankeData(max_poly_deg, n_data, data_dim = data_dim, add_noise = noise, test_size = test_size, set_seed=True)

    # XXX: check how solutions diverges

# =============== Boots ===============
    ra = resampling.ResamplingAnalysis(f)
    regression_methods = ['ols_own', 'ols_skl']
    score_list = ['mse', 'bias', 'variance']
    n_resamples = 100
    boots_scores = ra.bootstrap_loop(regression_methods = regression_methods,
            n_resamples = n_resamples, resample_dataset='train', lamb = 0.0001, predict_dataset='train')
    resampling.plot_bias_variance_tradeoff(boots_scores, score_list=score_list, 
            title = f'Bootstrap with n resamples = {n_resamples} and n datapoints = {n_data*n_data}')


