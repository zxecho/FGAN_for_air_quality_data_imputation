import numpy as np


def create_test_datasets():
    test_shape = (20, 20)
    a = np.ones(shape=test_shape)
    print('a:\n', a)

    line_miss_num = 2
    row_miss_num = 2
    p_miss = 0.2

    # 行缺失
    while True:
        rand_m1 = np.random.randint(0, a.shape[0], 2)
        u = np.unique(rand_m1)
        if len(u) == 2:
            break
    print('rand_m1: ', rand_m1)
    m1 = np.ones(shape=test_shape)
    line_missing_p_num = line_miss_num * a.shape[1]       # 计算出行缺失的数量

    # 列缺失
    m2 = np.ones(shape=test_shape)
    rand_m2_v = np.random.randint(1, a.shape[1], 2)
    rand_m2_l = np.random.randint(1, a.shape[0], 2)
    missing_len = np.random.randint(2, 6, 2)
    # 计算出列缺失的数量
    row_missing_p_num = 0
    for i in range(len(missing_len)):
        row_missing_p_num += missing_len[i]

    # 点缺失
    m3 = np.ones(shape=test_shape)
    miss_num = p_miss * a.shape[0] * a.shape[1]
    miss_p_num = miss_num - row_missing_p_num - line_missing_p_num
    prob_missing_p = miss_p_num / (a.shape[0] * a.shape[1])
    Missing = np.zeros((a.shape[0], a.shape[1]))
    p_miss_vec = prob_missing_p * np.ones((a.shape[0], 1))
    for i in range(a.shape[0]):
        A = np.random.uniform(0., 1., size=[len(a), ])
        B = A > p_miss_vec[i]
        Missing[:, i] = 1. * B
    print('m3: \n', Missing)

    m1[rand_m1, :] = m1[rand_m1, :] * 0     # 行缺失赋值
    print('m1: \n', m1)
    Missing[rand_m1, :] = Missing[rand_m1, :] * 0     # 行缺失赋值

    # 列缺失赋值
    for i in range(2):
        m2[rand_m2_l[i]: rand_m2_l[i]+missing_len[i], rand_m2_v[i]] = m2[rand_m2_l[i]: rand_m2_l[i]+missing_len[i], rand_m2_v[i]] * 0
    print('m2: \n', m2)
    m2 = Missing
    for i in range(2):
        m2[rand_m2_l[i]: rand_m2_l[i]+missing_len[i], rand_m2_v[i]] = m2[rand_m2_l[i]: rand_m2_l[i]+missing_len[i], rand_m2_v[i]] * 0
    print('Missing: \n', Missing)
    real_missing_num = 1 - Missing
    real_missing_num = real_missing_num.sum()
    print('real missing num :', real_missing_num, 'expect missing num: ', miss_num)


def plot_EM_results():
    # 绘制测试结果图像
    from param_options import args_parser
    from plot_indexes_resluts import plot_indicator_results
    from util_tools import mkdir

    args = args_parser()
    args.dataset_path = './constructed_datasets_6_30(1)/'
    exp_name = 'EM'
    result_save_file = './results/' + exp_name + '/'
    plots_file = './plot_results/' + exp_name + '/'

    print('====================== Save every component indicator result ========================')
    for indicator in ['rmse', 'd2', 'r2']:
        print('{} results'.format(indicator))
        fig_save_path = plots_file + indicator + '/'
        mkdir(fig_save_path)
        for component in args.select_dim:
            results_logdir = [result_save_file + indicator + '/']
            plot_indicator_results(results_logdir, fig_save_path, component, xaxis='station', leg=['EM'])

    print(">>> Finished save resluts figures!")


def soft(ratios):
    weights_array = np.array(ratios)
    soft_v = []
    for r in ratios:
        s = np.exp(r) / np.exp(weights_array).sum()
        soft_v.append(s)

    return soft_v


def plot_all_rmse_test():
    from param_options import args_parser
    from plot_indexes_resluts import plot_indicator_results

    args = args_parser()
    # 做实验
    exp_time = 5
    exp_name = 'FedWeightAvg(load_numbers_soft)_(5-1)_dn(10)_100_32(lrS_Adam_Fed_0.001_100_0.001_100_bs_128)_32'
    result_save_file = './results/' + exp_name + '/'
    indicator = 'all_rmse'

    # 绘制测试结果图像
    print('====================== Save every component indicator result ========================')

    results_logdir = [result_save_file + indicator + '/' + model_name + '/' for model_name in ['fed', 'idpt']]
    fig_save_path = result_save_file + indicator + '/'
    plot_indicator_results(results_logdir, fig_save_path, 'all_rmse')


def plot_datasets():
    # a = np.array([[1, 2, 3], [6, 9, 7]])
    # b = np.array([[7, 4, 6], [1, 6, 7]])
    # c = np.array([[6, 7, 1], [6, 2, 5]])
    # l = [a, b, c]
    # d = np.array(l)
    # print('d: ', d)

    from param_options import args_parser
    from plot_indexes_resluts import plot_all_datasets
    from loadDatasets import get_saved_datasets

    args = args_parser()
    args.dataset_path = './constructed_datasets_6_dn(10_SI2)/'
    all_station_datasets = get_saved_datasets(args)
    plot_all_datasets(all_station_datasets, args.select_dim, args.selected_stations, args.dataset_path)


def barchart_test():
    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    men_means = [20, 35, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]
    men_std = [2, 3, 4, 1, 2]
    women_std = [3, 5, 2, 3, 3]
    width = 0.35  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(labels, men_means, width, yerr=men_std, label='Men')
    ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
           label='Women')

    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # fig = plt.figure()
    #
    # plot_EM_results()

    # a = [1.0, 1.0, 1.0, 1.0, 1.0, 0.2]
    # r = soft(a)
    # print(r)

    # plot_all_rmse_test()
    # plot_datasets()
    # barchart_test()
    fig, ax = plt.subplots()
