import numpy as np
import pandas as pd
from functools import reduce

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline

from Fed_gain_test_run import plot_indicator_avg_results_m
# from LoadandCreateDatasets import get_saved_datasets
from CreateAndLoadDatasets import get_saved_datasets
from util_tools import compute_rmse, compute_d2, compute_r2, mkdir, save2csv, compute_avg_of_data_in_file, \
    save_as_csv
from util_tools import compute_all_rmse
from plot_indexes_resluts import plot_indicator_results


def impute_em(dataset, save_path='', exp_num=None, max_iter=3000, eps=1e-08):
    """(np.array, int, number) -> {str: np.array or int}

    Precondition: max_iter >= 1 and eps > 0

    Return the dictionary with five keys where:
    - Key 'mu' stores the mean estimate of the imputed data.
    - Key 'Sigma' stores the variance estimate of the imputed data.
    - Key 'X_imputed' stores the imputed data that is mutated from X using
      the EM algorithm.
    - Key 'C' stores the np.array that specifies the original missing entries
      of X.
    - Key 'iteration' stores the number of iteration used to compute
      'X_imputed' based on max_iter and eps specified.
    """
    Dim = dataset['d']
    trainX = dataset['train_x']
    testX = dataset['test_x']
    trainM = dataset['train_m']
    testM = dataset['test_m']
    # Train_No = dataset['train_no']
    # Test_No = dataset['test_no']

    X = testX.copy()
    Mask = testM

    X[Mask == 0] = np.nan
    nr, nc = X.shape
    C = np.isnan(X) == False

    # Collect M_i and O_i's
    one_to_nc = np.arange(1, nc + 1, step=1)
    M = one_to_nc * (C == False) - 1
    O = one_to_nc * C - 1

    # Generate Mu_0 and Sigma_0
    Mu = np.nanmean(X, axis=0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    S = np.cov(X[observed_rows,].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis=0))

    # Start updating
    Mu_tilde, S_tilde = {}, {}
    X_tilde = X.copy()
    no_conv = True
    iteration = 0
    while no_conv and iteration < max_iter:
        for i in range(nr):
            S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)
            if set(O[i,]) != set(one_to_nc - 1):  # missing component exists
                M_i, O_i = M[i, ][M[i, ] != -1], O[i, ][O[i, ] != -1]
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)]
                Mu_tilde[i] = Mu[np.ix_(M_i)] + S_MO @ np.linalg.inv(S_OO) @ (X_tilde[i, O_i] - Mu[np.ix_(O_i)])
                X_tilde[i, M_i] = Mu_tilde[i]
                S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O
        Mu_new = np.mean(X_tilde, axis=0)
        S_new = np.cov(X_tilde.T, bias=1) + reduce(np.add, S_tilde.values()) / nr
        no_conv = \
            np.linalg.norm(Mu - Mu_new) >= eps or \
            np.linalg.norm(S - S_new, ord=2) >= eps
        Mu = Mu_new
        S = S_new
        iteration += 1

    result = {
        'mu': Mu,
        'Sigma': S,
        'X_imputed': X_tilde,
        'C': C,
        'iteration': iteration
    }
    print('===================== EM algo station Test =======================')

    rmse_on_test_results = []
    d2_on_test_results = []
    r2_on_test_results = []
    for i in range(len(args.select_dim)):
        rmse = compute_rmse(testX[:, i], X_tilde[:, i])
        d2 = compute_d2(testX[:, i], X_tilde[:, i])
        r2 = compute_r2(testX[:, i], X_tilde[:, i])
        rmse_on_test_results.append(rmse)
        d2_on_test_results.append(d2)
        r2_on_test_results.append(r2)

    # 保存到本地
    # 选择的站点
    stations = args.selected_stations
    fed_save_csv_pt = save_path + 'rmse/EM_rmse_test_results_' + str(exp_num) + '.csv'
    save2csv(fed_save_csv_pt, rmse_on_test_results, stations, args.select_dim)
    # save_as_csv(fed_save_csv_pt, fed_mse_on_test_results, stations, 'MSE')
    fed_save_csv_pt = save_path + 'd2/EM_d2_test_results_' + str(exp_num) + '.csv'
    # save_as_csv(fed_save_csv_pt, fed_d2_on_test_results, stations, 'D2')
    save2csv(fed_save_csv_pt, d2_on_test_results, stations, args.select_dim)
    fed_save_csv_pt = save_path + 'r2/EM_r2_test_results_' + str(exp_num) + '.csv'
    # save_as_csv(fed_save_csv_pt, fed_r2_on_test_results, stations, 'R2')
    save2csv(fed_save_csv_pt, r2_on_test_results, stations, args.select_dim)

    return result


def Shrinkage_operator(tau, x):
    # shrinkage operator
    s = np.sign(x) * max(np.max(abs(x) - tau), 0)
    return s


def Do(tau, data):
    # shrinkage operator for singular values
    U, s, V = np.linalg.svd(data, full_matrices=False)
    So = Shrinkage_operator(tau, s)
    d = np.dot(U, np.dot(np.diag(So), V))
    return d


def Low_rank_completion(dataset, l, mu, initial=0, tol=1e-6, max_iter=2000):
    Dim = dataset['d']
    trainX = dataset['train_x']
    testX = dataset['test_x']
    trainM = dataset['train_m']
    testM = dataset['test_m']
    # Train_No = dataset['train_no']
    # Test_No = dataset['test_no']

    X = testX.copy()

    Mask = testM
    X[Mask == 0] = np.nan

    M, N = X.shape[0], X.shape[1]
    ERR = []
    K = []

    # l = 1 / np.sqrt(max(M, N))
    # mu = 10 * l

    unobserved = np.isnan(X)
    X[unobserved] = initial
    normX = np.linalg.norm(X)

    L = np.zeros(X.shape)
    S = np.zeros(X.shape)
    Y = np.zeros(X.shape)

    for i in range(max_iter):
        L = Do(1 / mu, X - S + (1 / mu) * Y)

        S = Shrinkage_operator(l / mu, X - L + (1 / mu)*Y)
        Z = X - L - S
        Z[unobserved] = 0
        Y = Y + mu * Z

        err = np.linalg.norm(Z) / normX

        if err.all() < tol:
            break

    print('Finishing Low rank completion!  Iter" {}'.format(i))

    MissX = (1 - Mask) * testX
    impute_x = (1 - Mask) * L
    print('>>> testX: \n', MissX)
    print('### impute x: \n', impute_x)

    print('//')

    return impute_x


def pandas_linear_methods(dataset):
    Dim = dataset['d']
    trainX = dataset['train_x']
    testX = dataset['test_x']
    trainM = dataset['train_m']
    testM = dataset['test_m']
    # Train_No = dataset['train_no']
    # Test_No = dataset['test_no']

    X = testX.copy()

    Mask = testM
    X[Mask == 0] = np.nan

    # 将array转成Dataframe
    X = pd.DataFrame(X)

    # 线性插值
    print('=======================  Linear interpolate  =========================')
    linear_intp = X.interpolate(method='linear')

    MissX = (1 - Mask) * testX
    linear_intp[linear_intp.isnull()] = 0
    impute_x = (1 - Mask) * linear_intp.values
    print('>>> testX: \n', MissX)
    print('### impute x: \n', impute_x)

    all_rmse = compute_all_rmse(MissX, impute_x, testM)

    return all_rmse


def pandas_cubic_method(dataset):
    Dim = dataset['d']
    trainX = dataset['train_x']
    testX = dataset['test_x']
    trainM = dataset['train_m']
    testM = dataset['test_m']
    # Train_No = dataset['train_no']
    # Test_No = dataset['test_no']

    X = testX.copy()

    Mask = testM
    X[Mask == 0] = np.nan

    # 将array转成Dataframe
    X = pd.DataFrame(X)
    # Cubic插值
    print('=======================  Cubic interpolate  =========================')
    linear_intp = X.interpolate(method='cubic', limit_direction='both', order=2)

    MissX = (1 - Mask) * testX
    linear_intp[linear_intp.isnull()] = 0
    impute_x = (1 - Mask) * linear_intp.values
    print('>>> testX: \n', MissX)
    print('### impute x: \n', impute_x)

    all_rmse = compute_all_rmse(MissX, impute_x, testM)

    return all_rmse


def pandas_spline_method(dataset):
    Dim = dataset['d']
    trainX = dataset['train_x']
    testX = dataset['test_x']
    trainM = dataset['train_m']
    testM = dataset['test_m']
    # Train_No = dataset['train_no']
    # Test_No = dataset['test_no']

    X = testX.copy()

    Mask = testM
    X[Mask == 0] = np.nan

    # 将array转成Dataframe
    X = pd.DataFrame(X)
    # Spline插值
    print('=======================  Spline interpolate  =========================')
    linear_intp = X.interpolate(method='spline', limit_direction='both', order=2)

    MissX = (1 - Mask) * testX
    impute_x = (1 - Mask) * linear_intp.values

    a_rmse = compute_all_rmse(MissX, impute_x, testM)

    return a_rmse


def pandas_nearest_method(dataset):
    Dim = dataset['d']
    trainX = dataset['train_x']
    testX = dataset['test_x']
    trainM = dataset['train_m']
    testM = dataset['test_m']
    # Train_No = dataset['train_no']
    # Test_No = dataset['test_no']

    X = testX.copy()

    Mask = testM
    X[Mask == 0] = np.nan

    # 将array转成Dataframe
    X = pd.DataFrame(X)
    # Nearest插值
    print('=======================  Nearest interpolate  =========================')
    linear_intp = X.interpolate(method='nearest', limit_direction='both')

    MissX = (1 - Mask) * testX
    linear_intp[linear_intp.isnull()] = 0
    impute_x = (1 - Mask) * linear_intp.values
    print('>>> testX: \n', MissX)
    print('### impute x: \n', impute_x)

    a_rmse = compute_all_rmse(MissX, impute_x, testM)

    return a_rmse


def MultiIterBayesian(dataset):

    Dim = dataset['d']
    trainX = dataset['train_x']
    testX = dataset['test_x']
    trainM = dataset['train_m']
    testM = dataset['test_m']
    # Train_No = dataset['train_no']
    # Test_No = dataset['test_no']

    test_X = testX.copy()
    train_X = trainX.copy()

    train_X[trainM == 0] = np.nan
    test_X[testM == 0] = np.nan

    # Bayesian imputation
    br_estimator = BayesianRidge()

    by_imp = IterativeImputer(random_state=0, estimator=br_estimator)
    by_imp.fit(train_X)

    imputed_X = by_imp.transform(test_X)

    print('>>>BayesianRidge IterativeImputer result: \n')
    print(imputed_X)

    _all_rmse = compute_rmse(testX, imputed_X, testM)

    print('>>>all_rmse', _all_rmse)

    return _all_rmse


def MultiIterTrees(dataset):
    from sklearn.impute import IterativeImputer

    Dim = dataset['d']
    trainX = dataset['train_x']
    testX = dataset['test_x']
    trainM = dataset['train_m']
    testM = dataset['test_m']
    # Train_No = dataset['train_no']
    # Test_No = dataset['test_no']

    test_X = testX.copy()
    train_X = trainX.copy()

    train_X[trainM == 0] = np.nan
    test_X[testM == 0] = np.nan

    # Bayesian imputation
    etr_estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)

    etr_imp = IterativeImputer(random_state=0, estimator=etr_estimator)
    etr_imp.fit(train_X)

    imputed_X = etr_imp.transform(test_X)

    print('>>>ExtraTreesRegressor IterativeImputer result: \n')
    print(imputed_X)

    _all_rmse = compute_rmse(testX, imputed_X, testM)

    print('>>>all_rmse', _all_rmse)

    return _all_rmse


if __name__ == '__main__':
    from param_options import args_parser

    args = args_parser()

    dataset_name = 'one_mi((A5)_1)'

    results_saved_file = 'otherAL_results'
    results_plot_file = 'plot_otherAL_results'
    # method_name = 'MultiBayesian'
    method_name = 'Cubic'      # Linear, Spline, KNN, Cubic, MultiBayesian, RandomTrees

    # exp_name = 'Multivariate_imputation_BayesianRidge_dn({})'.format(dataset_number)
    # args.dataset_path = './constructed_datasets/6_dn({})/0/'.format(dataset_number)

    # 做实验
    exp_total_time = 1
    cross_validation_sets = 5

    indicator_list = ['rmse', 'd2', 'r2', 'all_rmse']

    # params_test_list = ['Spline', 'MultiBayesian']      # Linear, Knn
    params_test_list = [5, 10, 15, 20, 25, 30, 35, 40]
    test_param_name = 'missing_ratio'       # method_name, missing_ratio

    for param in params_test_list:

        print('**  {} params test: {}  **'.format(test_param_name, param))
        dataset_name = 'one_mi_v1((A{})_1r)'.format(param)
        # dataset_name = 'one_mi((A5_A20_A30)_111)'
        # method_name = param
        leg = [method_name]
        exp_name = '{}_{}_T1'.format(method_name, dataset_name)
        for i in range(cross_validation_sets):
            print('============= Start training at datasets {} =============='.format(i))

            args.dataset_path = './constructed_datasets/{}/{}/'.format(dataset_name, i)
            station_datasets = get_saved_datasets(args)

            # 用于统计各种指标，建立相对应的文件夹
            result_save_file = './{}/'.format(results_saved_file) + exp_name + '/datasets_{}/'.format(i)

            # result_save_file = './{}/'.format(results_saved_file) + exp_name + '/'

            # 用于统计各种指标，建立相对应的文件夹
            for index in indicator_list:
                test_result_save_path = result_save_file + index + '/'
                mkdir(test_result_save_path)

            for exp_t in range(exp_total_time):
                all_rmse_on_test_results = []  # 用于存储每个数据集测试结果
                # 对每个数据集的每个站点数据进行计算
                for dataset, station in zip(station_datasets, args.selected_stations):
                    # r = Low_rank_completion(dataset, l=1, mu=0.02, initial=0)   # LRC method
                    # Linear, Spline, Cubic, MultiBayesian, RandomTrees
                    if method_name == 'Linear':
                        a_rmse = pandas_linear_methods(dataset)     # Linear
                    elif method_name == 'Spline':
                        a_rmse = pandas_spline_method(dataset)   # Spline
                    elif method_name == 'KNN':
                        a_rmse = pandas_nearest_method(dataset)     # KNN
                    elif method_name == 'Cubic':
                        a_rmse = pandas_cubic_method(dataset)     # Cubic
                    elif method_name == 'MultiBayesian':
                        a_rmse = MultiIterBayesian(dataset)     # MultiBayesian
                    elif method_name == 'RandomTrees':
                        a_rmse = MultiIterTrees(dataset)        # RandomTree
                    all_rmse_on_test_results.append(a_rmse)
                save_csv_pt = result_save_file + 'all_rmse/Time_{}_allrmse_test_results'.format(exp_t) + '.csv'
                save_as_csv(save_csv_pt, all_rmse_on_test_results, 'all_rmse')
        print('>>> Finished model evaluate on Test datasets!')

        result_save_root = './{}/'.format(results_saved_file) + exp_name + '/'
        plots_save_root = './{}/'.format(results_plot_file) + exp_name + '/'
        indicator_name = 'all_rmse'

        # 建立保存结果的文件夹
        indicator_avg_results_csv_save_fpth = result_save_root + 'avg_' + indicator_name + '/'
        for mode in leg:
            mkdir(indicator_avg_results_csv_save_fpth + mode + '/')

        # # 计算每个数据集的几次实验的均值
        for c in range(cross_validation_sets):
            results_logdir = [result_save_root + 'datasets_' + str(c) + '/' + indicator_name + '/']

            compute_avg_of_data_in_file(args, c, results_logdir, indicator_avg_results_csv_save_fpth,
                                        indicator_name, leg)

        # 绘制测试结果图像
        print('====================== Save every component indicator result ========================')

        print('{} results'.format(indicator_name))

        results_logdir = [result_save_root + 'avg_' + indicator_name + '/' + mode_name + '/' for mode_name in leg]
        fig_save_path = plots_save_root + indicator_name + '/'
        csv_save_fpth = result_save_root + 'avg_' + indicator_name + '/'
        mkdir(fig_save_path)

        # 将每个数据集计算的均值再计算总的均值和绘制方差均值线图
        plot_indicator_avg_results_m(results_logdir, fig_save_path, 'station', indicator_name, csv_save_fpth, leg)
        # plot_indicator_results(results_logdir, fig_save_path, indicator_name)

        print(">>> Finished save resluts figures!")

        # 绘制测试结果图像
        # plots_file = './plot_results_v2/' + exp_name + '/'
        #
        # print('====================== Save every component indicator result ========================')
        # for indicator in indicator_list:
        #     print('{} results'.format(indicator))
        #     fig_save_path = plots_file + indicator + '/'
        #     mkdir(fig_save_path)
        #     for component in args.select_dim:
        #         results_logdir = [result_save_file + indicator + '/']
        #         plot_indicator_results(results_logdir, fig_save_path, component, xaxis='station', leg=leg)
        #
        # print(">>> Finished save resluts figures!")
