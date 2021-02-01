import numpy as np
import pandas as pd
from functools import reduce

# from LoadandCreateDatasets import get_saved_datasets
from CreateAndLoadDatasets import get_saved_datasets
from util_tools import compute_rmse, compute_d2, compute_r2, compute_all_rmse
from util_tools import mkdir, save2csv, save_as_csv
from util_tools import compute_avg_of_data_in_file
from plot_indexes_resluts import plot_indicator_results
from Fed_gain_test_run import compute_indicator_results, plot_indicator_avg_results_m


def impute_em(dataset, save_path='', exp_num=None, max_iter=100, eps=1e-08):
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
        for j in range(nr):
            S_tilde[j] = np.zeros(nc ** 2).reshape(nc, nc)
            if set(O[j, ]) != set(one_to_nc - 1):  # missing component exists
                M_i, O_i = M[j, ][M[j, ] != -1], O[j, ][O[j, ] != -1]
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)]
                Mu_tilde[j] = Mu[np.ix_(M_i)] + S_MO @ np.linalg.inv(S_OO) @ (X_tilde[j, O_i] - Mu[np.ix_(O_i)])
                X_tilde[j, M_i] = Mu_tilde[j]
                S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                S_tilde[j][np.ix_(M_i, M_i)] = S_MM_O
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
        'X': testX,
        'X_imputed': X_tilde,
        'C': C,
        'iteration': iteration
    }
    print('===================== EM algo station Test =======================')

    rmse_list = []
    d2_list = []
    r2_list = []
    for j in range(len(args.select_dim)):
        _rmse = compute_rmse(testX[:, j], X_tilde[:, j], testM[:, j])
        _d2 = compute_d2(testX[:, j], X_tilde[:, j], testM[:, j])
        _r2 = compute_r2(testX[:, j], X_tilde[:, j])
        rmse_list.append(_rmse)
        d2_list.append(_d2)
        r2_list.append(_r2)

    # 保存all_rmse
    # _all_rmse = compute_rmse(testX, X_tilde, testM)
    _all_rmse = compute_all_rmse(testX, X_tilde, testM)

    return result, rmse_list, d2_list, r2_list, _all_rmse


if __name__ == '__main__':
    # 做实验
    # 绘制测试结果图像
    from param_options import args_parser

    args = args_parser()

    exp_total_time = 5
    cross_validation_sets_num = 5
    results_saved_file = 'otherAL_results'
    plot_saved_file = 'plot_otherAL_results'

    indicator_list = ['rmse', 'd2', 'r2', 'all_rmse']

    # params_test_list = [5, 10, 15, 20, 25, 30, 35, 40]
    # test_param_name = 'missing_ratio'

    params_test_list = [10]
    test_param_name = 'alpha'

    # dataset_number = '30'
    # exp_name = 'EM_6_dn({})'.format(dataset_number)

    for param in params_test_list:

        print('**  {} params test: {}  **'.format(test_param_name, param))
        # dataset_number = 'one_mi_v1((A{})_1r)'.format(param)
        dataset_number = 'one_mi((A5_A20_A30)_111)'
        exp_name = 'EM_{}_lastest_1'.format(dataset_number)

        result_save_file = './{}/'.format(results_saved_file) + exp_name + '/'

        # 用于统计各种指标，建立相对应的文件夹
        for ind in indicator_list:
            test_result_save_path = result_save_file + ind + '/'
            mkdir(test_result_save_path)

        for i in range(cross_validation_sets_num):
            print('============= Start training at datasets {} =============='.format(i))
            # 用于统计各种指标，建立相对应的文件夹
            # result_save_file = './{}/'.format(results_saved_file) + exp_name + '/datasets_{}/'.format(i)
            # plots_file = './plot_results/' + exp_name + '/datasets_{}/'.format(i)

            # 当前数据集
            args.dataset_path = './constructed_datasets/{}/{}/'.format(dataset_number, i)
            # 载入数据
            station_datasets = get_saved_datasets(args)

            rmse_on_test_results = []
            d2_on_test_results = []
            r2_on_test_results = []
            all_rmse_on_test_results = []
            for dataset, station in zip(station_datasets, args.selected_stations):
                r, rmse, d2, r2, all_rmse = impute_em(dataset, result_save_file, )
                impute_r = r['X_imputed']
                source_x = r['X']
                rmse_on_test_results.append(rmse)
                d2_on_test_results.append(d2)
                r2_on_test_results.append(r2)
                all_rmse_on_test_results.append(all_rmse)
                print('- Station {} complete federated evaluation!\n'.format(station))
            save_path = result_save_file
            # 选择的站点
            stations = args.selected_stations
            fed_save_csv_pt = save_path + 'rmse/Dataset_{}_EM_rmse_test_results'.format(i) + '.csv'
            save2csv(fed_save_csv_pt, rmse_on_test_results, stations, args.select_dim)
            # save_as_csv(fed_save_csv_pt, fed_mse_on_test_results, stations, 'MSE')
            fed_save_csv_pt = save_path + 'd2/Dataset_{}_EM_d2_test_results'.format(i) + '.csv'
            # save_as_csv(fed_save_csv_pt, fed_d2_on_test_results, stations, 'D2')
            save2csv(fed_save_csv_pt, d2_on_test_results, stations, args.select_dim)
            fed_save_csv_pt = save_path + 'r2/Dataset_{}_EM_r2_test_results'.format(i) + '.csv'
            # save_as_csv(fed_save_csv_pt, fed_r2_on_test_results, stations, 'R2')
            save2csv(fed_save_csv_pt, r2_on_test_results, stations, args.select_dim)
            save_csv_pt = result_save_file + 'all_rmse/Dataset_{}_EM_allrmse_test_results'.format(i) + '.csv'
            save_as_csv(save_csv_pt, all_rmse_on_test_results, 'all_rmse')

        print('>>> Finished EM model evaluate on Test datasets!')

        print('====================== Save every component indicator result ========================')

        result_save_root = './{}/'.format(results_saved_file) + exp_name + '/'
        plots_save_root = './{}/'.format(plot_saved_file) + exp_name + '/'
        indicator_name = 'all_rmse'
        leg = ['EM']

        # 建立保存结果的文件夹
        # indicator_results_csv_save_fpth = result_save_root + indicator_name + '/'
        # for mode in leg:
        #     mkdir(indicator_results_csv_save_fpth+mode+'/')

        # 计算每个数据集的几次实验的均值
        # for c in range(cross_validation_sets_num):
        #     results_logdir = [result_save_root + indicator_name + '/']
        #
        #     compute_avg_of_data_in_file(args, c, results_logdir, indicator_results_csv_save_fpth,
        #                                 indicator_name, leg)

        # 绘制测试结果图像
        print('====================== Save every component indicator result ========================')

        print('{} results'.format(indicator_name))

        results_logdir = [result_save_root + indicator_name + '/']
        fig_save_path = plots_save_root + indicator_name + '/'
        csv_save_fpth = result_save_root
        mkdir(fig_save_path)

        # 计算几次实验的最后的平均值并画出均值和方差曲线图
        plot_indicator_avg_results_m(results_logdir, fig_save_path, 'station', indicator_name, csv_save_fpth, leg)

        print(">>> Finished save resluts figures!")
